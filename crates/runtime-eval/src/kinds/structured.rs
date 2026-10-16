//! `structured.v1`: structured output, one row per request of the 12 cases
//! (JSON schema, choice, regex and tool outputs, each validated locally),
//! with a summary recomputed from the rows
//! (docs/check-kinds.md#structured-structuredv1, `tools/qual/structured.py`).

use std::cmp::Ordering;

use serde_json::{Map, Value, json};

use crate::capture::{self, flag, option, receipt, thinking_kwargs};
use crate::declaration::Check;
use crate::http::{Event, HttpError};
use crate::json::{python_cmp, python_eq, python_float_repr, python_round, truthy};
use crate::kind::{Capture, Kind, KindError, KindInfo, SHARED_KINDS, default_score, natural};
use crate::kinds::{applied_constants, constant, get, invalid, require};
use crate::pyjson::{self, LoadError, Object};
use crate::repr::{self, type_name};

pub struct Structured;

/// The structured.v1 cases in the order `structured.py` runs them
/// (`kinds.STRUCTURED_CASES`).
pub const STRUCTURED_CASES: [&str; 12] = [
    "chat_json_object",
    "chat_json_schema_strict",
    "chat_json_schema_loose",
    "chat_so_json",
    "chat_so_choice",
    "chat_so_regex",
    "chat_so_json_object",
    "completion_so_json",
    "completion_json_schema",
    "tool_required",
    "tool_named",
    "reasoning_json_schema",
];
/// A row's statuses, in the order the summary counts them
/// (`kinds.STRUCTURED_STATUSES`).
const STATUSES: [&str; 4] = ["passed", "failed", "refused", "truncated"];

impl Kind for Structured {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[5]
    }

    /// `structured.main`: the 12 cases (or `--cases`), each `--repeat`
    /// times and with `--stream` streamed too, validated locally.
    fn capture(&self, capture: &mut Capture<'_>) -> Result<Value, KindError> {
        capture_structured(capture)
    }

    /// Every structured.v1 case, `repeat` times, plain then streamed with
    /// `stream`, each row with a known status (`kinds.Structured.verify`).
    fn verify(
        &self,
        receipt: &Value,
        _reference: Option<&Value>,
        _check: &Check,
    ) -> Result<Vec<String>, KindError> {
        let settings = get(receipt, "settings").and_then(Value::as_object);
        let runs = match settings {
            Some(settings) => {
                let cases = match settings.get("cases").filter(|cases| truthy(cases)) {
                    Some(cases) => iterate(cases)?,
                    None => Vec::new(),
                };
                let every = cases.len() == STRUCTURED_CASES.len()
                    && cases
                        .iter()
                        .zip(STRUCTURED_CASES)
                        .all(|(case, name)| case.as_str() == Some(name));
                let repeat = settings
                    .get("repeat")
                    .filter(|repeat| natural(repeat))
                    .and_then(Value::as_u64)
                    .filter(|repeat| *repeat > 0);
                match (every, repeat, settings.get("stream")) {
                    (true, Some(repeat), Some(Value::Bool(stream))) => Some((repeat, *stream)),
                    _ => None,
                }
            }
            None => None,
        };
        require(runs.is_some(), || {
            "settings do not run every structured.v1 case".into()
        })?;
        let (repeat, stream) = runs.expect("checked");
        let per_case = u128::from(repeat) * if stream { 2 } else { 1 };
        let total = per_case * STRUCTURED_CASES.len() as u128;
        let rows = get(receipt, "rows").and_then(Value::as_array);
        let named = rows.is_some_and(|rows| {
            rows.len() as u128 == total
                && rows.iter().enumerate().all(|(index, row)| {
                    let case = STRUCTURED_CASES[(index as u128 / per_case) as usize];
                    let suffix = if stream && index % 2 == 1 {
                        "+stream"
                    } else {
                        ""
                    };
                    get(row, "name").and_then(Value::as_str) == Some(&format!("{case}{suffix}"))
                })
        });
        require(named, || {
            format!("the rows are not the {total} requests the settings run")
        })?;
        let known = rows.expect("checked").iter().all(|row| {
            get(row, "status")
                .and_then(Value::as_str)
                .is_some_and(|status| STATUSES.contains(&status))
        });
        require(known, || "a row has an unknown status".into())?;
        Ok(vec!["rows".into()])
    }

    /// Recomputes the overall and per-case summaries and `passed` from the
    /// rows (`kinds.Structured.score`).
    fn score(
        &self,
        receipt: &Value,
        _reference: Option<&Value>,
        check: &Check,
        mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        let mut metrics = default_score(self, receipt, check, mechanics)?;
        let Some(rows) = get(receipt, "rows") else {
            return Ok(metrics);
        };
        let rows = iterate(rows)?;
        metrics.insert("summary".into(), summary(&rows)?);
        let failed = rows.iter().any(|row| row["status"] == "failed");
        metrics.insert("passed".into(), json!(!failed));
        Ok(metrics)
    }
}

/// The overall and per-case summaries of `rows`, cases in the order they
/// first appear (`structured.main`, `kinds.Structured.score`).
fn summary(rows: &[Value]) -> Result<Value, KindError> {
    let mut cases: Vec<(&Value, Vec<&Value>)> = Vec::new();
    for row in rows {
        let name = item(row, "name")?;
        if matches!(name, Value::Array(_) | Value::Object(_)) {
            return Err(invalid(format!("unhashable type: '{}'", type_name(name))));
        }
        match cases.iter_mut().find(|(key, _)| python_eq(key, name)) {
            Some((_, group)) => group.push(row),
            None => cases.push((name, vec![row])),
        }
    }
    let all: Vec<&Value> = rows.iter().collect();
    let overall = summarize(&all)?;
    let mut by_case = Map::new();
    for (name, group) in cases {
        by_case.insert(key(name), summarize(&group)?);
    }
    Ok(json!({"overall": overall, "cases": by_case}))
}

/// The status counts, the validity rate over judged rows, latency p50 and
/// p90 and the mean tokens per second of `rows` (`structured.summarize`).
pub fn summarize(rows: &[&Value]) -> Result<Value, KindError> {
    let mut counts = [0u64; 4];
    for (count, status) in counts.iter_mut().zip(STATUSES) {
        for row in rows {
            if item(row, "status")? == status {
                *count += 1;
            }
        }
    }
    let judged = counts[0] + counts[1];
    let latencies: Vec<&Value> = rows
        .iter()
        .filter_map(|row| get(row, "latency_seconds"))
        .collect();
    let speeds: Vec<&Value> = rows
        .iter()
        .filter_map(|row| get(row, "tokens_per_second"))
        .filter(|speed| truthy(speed))
        .collect();
    let validity = (judged > 0).then(|| python_round(counts[0] as f64 / judged as f64, 4));
    let mean = match speeds.len() {
        0 => None,
        length => Some(python_round(mean(&speeds, length)?, 2)),
    };
    let mut summary = Map::new();
    for (count, status) in counts.iter().zip(STATUSES) {
        summary.insert(status.into(), json!(count));
    }
    summary.insert("validity_rate".into(), json!(validity));
    summary.insert("latency_p50_seconds".into(), percentile(&latencies, 0.5)?);
    summary.insert("latency_p90_seconds".into(), percentile(&latencies, 0.9)?);
    summary.insert("mean_tokens_per_second".into(), json!(mean));
    Ok(Value::Object(summary))
}

/// The value at `fraction` of the sorted `values`, `None` when there are
/// none (`structured.percentile`).
fn percentile(values: &[&Value], fraction: f64) -> Result<Value, KindError> {
    let ordered = sorted(values)?;
    Ok(match ordered.len() {
        0 => Value::Null,
        length => ordered[(length - 1).min((fraction * length as f64) as usize)].clone(),
    })
}

/// Python's `sorted` (stable) of numbers or of strings.
fn sorted<'a>(values: &[&'a Value]) -> Result<Vec<&'a Value>, KindError> {
    sortable(values)?;
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| ascending(a, b));
    Ok(ordered)
}

/// Refuses values Python cannot sort: two neighbours that are not both
/// numbers or both strings.
fn sortable(values: &[&Value]) -> Result<(), KindError> {
    let number = |value: &Value| matches!(value, Value::Number(_) | Value::Bool(_));
    let comparable =
        |a: &Value, b: &Value| (number(a) && number(b)) || (a.is_string() && b.is_string());
    match values.windows(2).find(|pair| !comparable(pair[0], pair[1])) {
        Some(pair) => Err(invalid(format!(
            "'<' not supported between instances of '{}' and '{}'",
            type_name(pair[1]),
            type_name(pair[0])
        ))),
        None => Ok(()),
    }
}

/// Python's order of two numbers or two strings (by code point).
fn ascending(a: &Value, b: &Value) -> Ordering {
    match (a.as_str(), b.as_str()) {
        (Some(a), Some(b)) => a.cmp(b),
        _ => python_cmp(a, b).unwrap_or(Ordering::Equal),
    }
}

/// `sum(values) / length` of truthy speeds, as Python 3.12 computes it:
/// exact while every value is an int (a bool counts as one), then floats
/// with Neumaier compensation, ints added to them uncompensated; an int
/// total divided correctly rounded.
fn mean(values: &[&Value], length: usize) -> Result<f64, KindError> {
    let mut total: i128 = 0;
    let mut float: Option<(f64, f64)> = None;
    for value in values {
        let unsupported = |left: &str| {
            invalid(format!(
                "unsupported operand type(s) for +: '{left}' and '{}'",
                type_name(value)
            ))
        };
        match (&mut float, value) {
            (None, Value::Bool(flag)) => total += i128::from(*flag),
            (None, Value::Number(number)) if !number.is_f64() => {
                total += number
                    .as_i64()
                    .map(i128::from)
                    .or_else(|| number.as_u64().map(i128::from))
                    .expect("a JSON int");
            }
            (None, Value::Number(number)) => {
                float = Some((total as f64 + number.as_f64().expect("a float"), 0.0));
            }
            (Some((sum, compensation)), Value::Number(number)) if number.is_f64() => {
                let x = number.as_f64().expect("a float");
                let t = *sum + x;
                if sum.abs() >= x.abs() {
                    *compensation += (*sum - t) + x;
                } else {
                    *compensation += (x - t) + *sum;
                }
                *sum = t;
            }
            (Some((sum, _)), Value::Number(_) | Value::Bool(_)) => {
                *sum += crate::json::python_number(value).expect("a number");
            }
            (None, _) => return Err(unsupported("int")),
            (Some(_), _) => return Err(unsupported("float")),
        }
    }
    Ok(match float {
        None => total as f64 / length as f64,
        Some((sum, compensation)) => {
            let sum = if compensation != 0.0 && compensation.is_finite() {
                sum + compensation
            } else {
                sum
            };
            sum / length as f64
        }
    })
}

/// The items of a JSON value as Python iterates it: a list's items, a
/// dict's keys, a string's characters.
fn iterate(value: &Value) -> Result<Vec<Value>, KindError> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Object(map) => Ok(map.keys().map(|key| json!(key)).collect()),
        Value::String(text) => Ok(text.chars().map(|character| json!(character)).collect()),
        other => Err(invalid(format!(
            "'{}' object is not iterable",
            type_name(other)
        ))),
    }
}

/// Python's `value[key]` of a JSON value, with Python's error.
fn item<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Fault> {
    match value {
        Value::Object(map) => map
            .get(key)
            .ok_or_else(|| Fault::caught("KeyError", format!("'{key}'"))),
        Value::Array(_) => Err(Fault::caught(
            "TypeError",
            "list indices must be integers or slices, not str",
        )),
        Value::String(_) => Err(Fault::caught(
            "TypeError",
            "string indices must be integers, not 'str'",
        )),
        other => Err(Fault::caught(
            "TypeError",
            format!("'{}' object is not subscriptable", type_name(other)),
        )),
    }
}

/// A row name as `json.dumps` writes it as an object key.
fn key(name: &Value) -> String {
    match name {
        Value::String(text) => text.clone(),
        Value::Null => "null".into(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) if number.is_f64() => {
            python_float_repr(number.as_f64().expect("a float"))
        }
        other => other.to_string(),
    }
}

/// How Python fails on a value: an error `structured.run_case` catches
/// (`ValueError`, `KeyError` or `TypeError`, written `<type>: <message>`
/// in the row), or one it does not, which stops the run before its
/// receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Fault {
    Caught { kind: &'static str, message: String },
    Uncaught(String),
}

impl Fault {
    fn caught(kind: &'static str, message: impl Into<String>) -> Self {
        Self::Caught {
            kind,
            message: message.into(),
        }
    }

    fn value(message: impl Into<String>) -> Self {
        Self::caught("ValueError", message)
    }

    fn type_error(message: impl Into<String>) -> Self {
        Self::caught("TypeError", message)
    }
}

/// Offline, any error fails the rescore with Python's `str(error)`.
impl From<Fault> for KindError {
    fn from(fault: Fault) -> Self {
        match fault {
            Fault::Caught { message, .. } | Fault::Uncaught(message) => invalid(message),
        }
    }
}

/// `structured.PERSON`.
fn person() -> Value {
    json!({
        "type": "object", "additionalProperties": false,
        "required": ["name", "age", "status", "kind", "address", "tags", "active", "nickname", "note"],
        "properties": {
            "name": {"type": "string", "maxLength": 40},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            "kind": {"const": "person"},
            "address": {"type": "object", "additionalProperties": false, "required": ["city", "zip"],
                        "properties": {"city": {"type": "string", "maxLength": 40},
                                       "zip": {"type": "string", "maxLength": 10}}},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 20},
                     "minItems": 1, "maxItems": 3},
            "active": {"type": "boolean"},
            "nickname": {"anyOf": [{"type": "string", "maxLength": 20}, {"type": "null"}]},
            "note": {"type": "null"},
        },
    })
}

/// `structured.PERSON_PROMPT`.
const PERSON_PROMPT: &str = "Return a JSON record for a fictional person named Ada, aged 36, living in Paris (zip 75001), with a status, a few tags, whether she is active, and an optional nickname.";
/// `structured.PHONE`.
const PHONE: &str = r"\d{3}-\d{4}";

/// `structured.CHOICES`.
fn choices() -> Value {
    json!(["positive", "negative", "neutral"])
}

/// `structured.TOOLS`.
fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "additionalProperties": false, "required": ["city"],
                           "properties": {"city": {"type": "string"},
                                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}},
        {"type": "function", "function": {
            "name": "get_time", "description": "Get the current time in a timezone.",
            "parameters": {"type": "object", "additionalProperties": false, "required": ["timezone"],
                           "properties": {"timezone": {"type": "string"}}}}},
    ])
}

/// The module constants a check may set (`checks.<id>.constants.structured`).
/// `PARAMETERS` and `TYPES` are computed when the module loads (from the
/// default `TOOLS`); a check that sets them is refused.
const CONSTANTS: [&str; 5] = ["PERSON", "PERSON_PROMPT", "CHOICES", "PHONE", "TOOLS"];

/// The module's constants as the check sets them.
struct Constants {
    person: Value,
    person_prompt: String,
    choices: Value,
    phone: String,
    tools: Value,
    /// `structured.PARAMETERS`: each default tool's parameters schema.
    parameters: Map<String, Value>,
}

impl Constants {
    fn of(check: &Check) -> Result<Self, KindError> {
        let applied: Vec<(&str, &str)> =
            CONSTANTS.iter().map(|name| ("structured", *name)).collect();
        applied_constants(check, &applied)?;
        let read = |name: &str, default: Value, fits: fn(&Value) -> bool, what: &str| {
            let value = constant(check, "structured", name, default);
            if fits(&value) {
                Ok(value)
            } else {
                Err(invalid(format!(
                    "checks.{}.constants.structured.{name} must be a {what}",
                    check.id
                )))
            }
        };
        let text = |value: Value| value.as_str().map(String::from).expect("checked");
        let parameters = tools()
            .as_array()
            .expect("a list")
            .iter()
            .map(|tool| {
                let function = &tool["function"];
                (
                    function["name"].as_str().expect("a name").to_owned(),
                    function["parameters"].clone(),
                )
            })
            .collect();
        Ok(Self {
            person: read("PERSON", person(), Value::is_object, "dict")?,
            person_prompt: text(read(
                "PERSON_PROMPT",
                json!(PERSON_PROMPT),
                Value::is_string,
                "str",
            )?),
            choices: read("CHOICES", choices(), Value::is_array, "list")?,
            phone: text(read("PHONE", json!(PHONE), Value::is_string, "str")?),
            tools: read("TOOLS", tools(), Value::is_array, "list")?,
            parameters,
        })
    }
}

/// How a case's output is validated (`structured.check_json`,
/// `check_choice`, `check_regex`, `check_tools`).
enum Validator {
    Json(Option<Value>),
    Choice,
    Regex,
    Tools(Option<&'static str>),
}

/// One case: its name, path, body without model or sampling fields, and
/// validator.
struct Case {
    name: &'static str,
    path: &'static str,
    body: Value,
    validator: Validator,
}

/// The 12 cases in order (`structured.cases`). `thinking` is the
/// frontend's switch, off then on.
/// `thinking` holds the answer-only and the thinking switches
/// (`common.answer_kwargs`, `thinking_kwargs(True)`); where `reasons` (the
/// frontend refuses disabled thinking) an answer-only case follows its
/// reasoning within `reasoning_max_tokens` (`structured.cases`).
fn cases(
    constants: &Constants,
    thinking: &[Value; 2],
    reasoning_max_tokens: i64,
    reasons: bool,
) -> Vec<Case> {
    let chat = |prompt: &str, switch: &Value, extra: Value| {
        let mut body = json!({"messages": [{"role": "user", "content": prompt}],
                              "chat_template_kwargs": switch});
        if let (Some(body), Value::Object(extra)) = (body.as_object_mut(), extra) {
            body.extend(extra);
            if reasons && switch == &thinking[0] && !body.contains_key("max_tokens") {
                body.insert("max_tokens".into(), json!(reasoning_max_tokens));
            }
        }
        body
    };
    let off = &thinking[0];
    let person = &constants.person;
    let prompt = constants.person_prompt.as_str();
    let schema_format = |strict: bool| {
        json!({"type": "json_schema",
               "json_schema": {"name": "person", "schema": person, "strict": strict}})
    };
    let completion_prompt = format!("{prompt}\nJSON:");
    let weather =
        "What's the weather in Paris right now, in celsius? Also, what time is it in Tokyo?";
    let (chat_path, completion_path) = ("/v1/chat/completions", "/v1/completions");
    let json_check = || Validator::Json(Some(person.clone()));
    let case = |name, path, body, validator| Case {
        name,
        path,
        body,
        validator,
    };
    vec![
        case(
            "chat_json_object",
            chat_path,
            chat(
                "Return a JSON object describing Paris with keys name and country.",
                off,
                json!({"response_format": {"type": "json_object"}}),
            ),
            Validator::Json(None),
        ),
        case(
            "chat_json_schema_strict",
            chat_path,
            chat(prompt, off, json!({"response_format": schema_format(true)})),
            json_check(),
        ),
        case(
            "chat_json_schema_loose",
            chat_path,
            chat(
                prompt,
                off,
                json!({"response_format": schema_format(false)}),
            ),
            json_check(),
        ),
        case(
            "chat_so_json",
            chat_path,
            chat(prompt, off, json!({"structured_outputs": {"json": person}})),
            json_check(),
        ),
        case(
            "chat_so_choice",
            chat_path,
            chat(
                "Classify the sentiment of: 'I love this phone.'",
                off,
                json!({"structured_outputs": {"choice": constants.choices}}),
            ),
            Validator::Choice,
        ),
        case(
            "chat_so_regex",
            chat_path,
            chat(
                "Give a phone number in the form 555-0123.",
                off,
                json!({"structured_outputs": {"regex": constants.phone}}),
            ),
            Validator::Regex,
        ),
        case(
            "chat_so_json_object",
            chat_path,
            chat(
                "Return only a JSON object with exactly two string fields: name (Paris) and country (France).",
                off,
                json!({"structured_outputs": {"json_object": true}}),
            ),
            Validator::Json(None),
        ),
        case(
            "completion_so_json",
            completion_path,
            json!({"prompt": completion_prompt, "structured_outputs": {"json": person}}),
            json_check(),
        ),
        case(
            "completion_json_schema",
            completion_path,
            json!({"prompt": completion_prompt, "response_format": schema_format(true)}),
            json_check(),
        ),
        case(
            "tool_required",
            chat_path,
            chat(
                weather,
                off,
                json!({"tools": constants.tools, "tool_choice": "required"}),
            ),
            Validator::Tools(None),
        ),
        case(
            "tool_named",
            chat_path,
            chat(
                weather,
                off,
                json!({"tools": constants.tools,
                       "tool_choice": {"type": "function", "function": {"name": "get_time"}}}),
            ),
            Validator::Tools(Some("get_time")),
        ),
        case(
            "reasoning_json_schema",
            chat_path,
            chat(
                prompt,
                &thinking[1],
                json!({"response_format": schema_format(true), "max_tokens": reasoning_max_tokens}),
            ),
            json_check(),
        ),
    ]
}

/// The settings `structured.main` reads from its arguments, with its
/// defaults.
struct Settings {
    cases: Vec<String>,
    repeat: i64,
    stream: bool,
    max_tokens: i64,
    reasoning_max_tokens: i64,
    temperature: f64,
    seed: Option<i64>,
}

impl Settings {
    fn of(check: &Check) -> Result<Self, KindError> {
        if flag(check, "--self-test") {
            return Err(invalid(
                "--self-test checks the validator and writes no receipt",
            ));
        }
        let cases = option(check, "--cases")
            .filter(|cases| !cases.is_empty())
            .map(|cases| cases.split(',').map(String::from).collect())
            .unwrap_or_else(|| STRUCTURED_CASES.iter().map(|name| (*name).into()).collect());
        Ok(Self {
            cases,
            repeat: number(check, "--repeat", "int")?.unwrap_or(1),
            stream: flag(check, "--stream"),
            max_tokens: number(check, "--max-tokens", "int")?.unwrap_or(512),
            reasoning_max_tokens: number(check, "--reasoning-max-tokens", "int")?.unwrap_or(4096),
            temperature: number(check, "--temperature", "float")?.unwrap_or(1.0),
            seed: number(check, "--seed", "int")?,
        })
    }
}

/// The number after `name` in the check's arguments, as argparse's `type`
/// (`kind`) reads it.
fn number<T: std::str::FromStr>(
    check: &Check,
    name: &str,
    kind: &str,
) -> Result<Option<T>, KindError> {
    option(check, name)
        .map(|value| {
            value.trim().parse().map_err(|_| {
                invalid(format!(
                    "argument {name}: invalid {kind} value: {}",
                    repr::repr(&json!(value))
                ))
            })
        })
        .transpose()
}

/// `structured.main`: each named case `repeat` times, plain then streamed
/// with `--stream`, then the receipt.
fn capture_structured(capture: &mut Capture<'_>) -> Result<Value, KindError> {
    let settings = Settings::of(capture.check)?;
    let constants = Constants::of(capture.check)?;
    let reasons = !capture::thinking_can_be_disabled(capture)?;
    let thinking = [
        capture::answer_kwargs(capture)?,
        thinking_kwargs(capture, true)?,
    ];
    let catalog = cases(
        &constants,
        &thinking,
        settings.reasoning_max_tokens,
        reasons,
    );
    let unknown: Vec<&String> = settings
        .cases
        .iter()
        .filter(|name| !catalog.iter().any(|case| case.name == name.as_str()))
        .collect();
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "unknown cases {}; choose from {}",
            repr::repr(&json!(unknown)),
            repr::repr(&json!(STRUCTURED_CASES))
        )));
    }
    if settings.repeat < 1 {
        return Err(invalid("repeat must be positive"));
    }
    let model = json!(capture.server.model);
    let mut rows = Vec::new();
    for name in &settings.cases {
        let case = catalog
            .iter()
            .find(|case| case.name == name.as_str())
            .expect("checked");
        let mut body = json!({"model": model, "temperature": settings.temperature,
                              "max_tokens": settings.max_tokens});
        if let (Some(body), Value::Object(fields)) = (body.as_object_mut(), &case.body) {
            body.extend(fields.clone());
        }
        if capture.mechanics {
            body["max_tokens"] = json!(8);
        }
        if let Some(seed) = settings.seed {
            body["seed"] = json!(seed);
        }
        for _ in 0..settings.repeat {
            for stream in if settings.stream {
                &[false, true][..]
            } else {
                &[false][..]
            } {
                rows.push(run_case(capture, case, &body, &constants, *stream)?);
            }
        }
    }
    let summary = summary(&rows)?;
    let errors: Vec<Value> = rows
        .iter()
        .filter(|row| row["status"] == "failed")
        .map(|row| json!({"name": row["name"], "error": row.get("error").unwrap_or(&json!(""))}))
        .collect();
    let skipped: &[&str] = if capture.mechanics {
        &["output validity (8-token outputs)"]
    } else {
        &[]
    };
    let result = json!({
        "passed": errors.is_empty(), "skipped": skipped, "errors": errors,
        "summary": summary, "rows": rows,
        "settings": {"cases": settings.cases, "repeat": settings.repeat, "stream": settings.stream,
                     "max_tokens": settings.max_tokens,
                     "reasoning_max_tokens": settings.reasoning_max_tokens,
                     "temperature": settings.temperature, "seed": settings.seed}});
    let Value::Object(result) = result else {
        unreachable!("a result is an object")
    };
    Ok(receipt("structured", capture, result))
}

/// One request of a case and its row (`structured.run_case`): HTTP 400 is
/// `refused` and any other error status `failed`; finish `length` is
/// `truncated`; otherwise the output is validated. An unreachable server
/// or a spent budget is the environment's.
fn run_case(
    capture: &mut Capture<'_>,
    case: &Case,
    body: &Value,
    constants: &Constants,
    stream: bool,
) -> Result<Value, KindError> {
    let chat = case.path.ends_with("/chat/completions");
    let name = if stream {
        format!("{}+stream", case.name)
    } else {
        case.name.to_owned()
    };
    let mut row = Map::new();
    row.insert("name".into(), json!(name));
    row.insert("stream".into(), json!(stream));
    let mut body = body.clone();
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    capture::remaining(capture)?;
    let client = &mut *capture.client;
    let sent = if stream {
        client
            .stream(case.path, &body)
            .map(|(events, latency)| (Err(events), latency))
    } else {
        client
            .json(case.path, Some(&body))
            .map(|(value, latency)| (Ok(value), latency))
    };
    let (value, latency) = match sent {
        Ok(sent) => sent,
        Err(error @ HttpError::Transport(_)) => return Err(capture::http_error(error)),
        Err(error) => {
            let text = error.to_string();
            let status = if text.starts_with("HTTP 400") {
                "refused"
            } else {
                "failed"
            };
            row.insert("status".into(), json!(status));
            row.insert("error".into(), json!(text));
            return Ok(Value::Object(row));
        }
    };
    let value = match value {
        Ok(value) => value,
        Err(events) => assemble(&events, chat)?,
    };
    let response = dict(&value)?;
    let nothing = Map::new();
    let choice = match response.get("choices").filter(|choices| truthy(choices)) {
        None => &nothing,
        Some(Value::Array(choices)) => dict(&choices[0])?,
        Some(other) => return Err(uncaught(other, "the choices")),
    };
    let usage = match response.get("usage").filter(|usage| truthy(usage)) {
        None => &nothing,
        Some(usage) => dict(usage)?,
    };
    let tokens = usage
        .get("completion_tokens")
        .cloned()
        .unwrap_or(Value::Null);
    let speed = if truthy(&tokens) && latency > 0.0 {
        let tokens = crate::json::python_number(&tokens).ok_or_else(|| {
            invalid(format!(
                "unsupported operand type(s) for /: '{}' and 'float'",
                type_name(&tokens)
            ))
        })?;
        json!(python_round(tokens / latency, 2))
    } else {
        Value::Null
    };
    row.insert("latency_seconds".into(), json!(python_round(latency, 4)));
    row.insert("completion_tokens".into(), tokens);
    row.insert("tokens_per_second".into(), speed);
    let finish = choice.get("finish_reason").cloned().unwrap_or(Value::Null);
    row.insert("finish_reason".into(), finish.clone());
    let empty = json!({});
    let message = choice
        .get("message")
        .filter(|message| truthy(message))
        .unwrap_or(&empty);
    let content = if chat {
        dict(message)?.get("content")
    } else {
        choice.get("text")
    };
    if finish == "length" {
        row.insert("status".into(), json!("truncated"));
        row.insert("output".into(), slice(content, -200)?);
        return Ok(Value::Object(row));
    }
    let output = content
        .filter(|content| truthy(content))
        .cloned()
        .unwrap_or_else(|| json!(""));
    match validate(&case.validator, &output, message, constants) {
        Ok(()) => {
            row.insert("status".into(), json!("passed"));
        }
        Err(Fault::Caught { kind, message }) => {
            row.insert("status".into(), json!("failed"));
            row.insert("error".into(), json!(format!("{kind}: {message}")));
            row.insert("output".into(), slice(content, 500)?);
        }
        Err(Fault::Uncaught(message)) => return Err(invalid(message)),
    }
    Ok(Value::Object(row))
}

/// An error Python does not catch, reading `what`.
fn uncaught(value: &Value, what: &str) -> KindError {
    invalid(format!(
        "structured.py cannot read {what} of type '{}'",
        type_name(value)
    ))
}

/// A response object Python reads with `.get`.
fn dict(value: &Value) -> Result<&Map<String, Value>, KindError> {
    value.as_object().ok_or_else(|| {
        invalid(format!(
            "'{}' object has no attribute 'get'",
            type_name(value)
        ))
    })
}

/// `(content or "")[-count:]` for a negative `count`, else
/// `(content or "")[:count]`: a string's characters or a list's items.
fn slice(content: Option<&Value>, count: i64) -> Result<Value, KindError> {
    let take = count.unsigned_abs() as usize;
    match content.filter(|content| truthy(content)) {
        None => Ok(json!("")),
        Some(Value::String(text)) => {
            let characters: Vec<char> = text.chars().collect();
            let range = if count < 0 {
                characters.len().saturating_sub(take)..characters.len()
            } else {
                0..take.min(characters.len())
            };
            Ok(json!(characters[range].iter().collect::<String>()))
        }
        Some(Value::Array(items)) => {
            let range = if count < 0 {
                items.len().saturating_sub(take)..items.len()
            } else {
                0..take.min(items.len())
            };
            Ok(Value::Array(items[range].to_vec()))
        }
        Some(other) => Err(uncaught(other, "the output")),
    }
}

/// A non-streamed response rebuilt from stream events
/// (`structured.assemble`): the content joined, each tool call's name and
/// arguments joined by index, the last finish reason and usage.
fn assemble(events: &[Event], chat: bool) -> Result<Value, KindError> {
    let mut content = String::new();
    let mut calls: Vec<(Value, Map<String, Value>)> = Vec::new();
    let mut finish = Value::Null;
    let mut usage = Value::Null;
    let text = |value: Option<&Value>| -> Result<String, KindError> {
        match value.filter(|value| truthy(value)) {
            None => Ok(String::new()),
            Some(Value::String(text)) => Ok(text.clone()),
            Some(other) => Err(uncaught(other, "streamed texts")),
        }
    };
    let list = |value: Option<&Value>| -> Result<Vec<Value>, KindError> {
        match value.filter(|value| truthy(value)) {
            None => Ok(Vec::new()),
            Some(Value::Array(items)) => Ok(items.clone()),
            Some(other) => Err(uncaught(other, "streamed lists")),
        }
    };
    for (_, event) in events {
        let event = dict(event)?;
        if let Some(value) = event.get("usage").filter(|usage| truthy(usage)) {
            usage = value.clone();
        }
        for choice in list(event.get("choices"))? {
            let choice = dict(&choice)?;
            if let Some(value) = choice.get("finish_reason").filter(|value| truthy(value)) {
                finish = value.clone();
            }
            let empty = Map::new();
            let delta = match (chat, choice.get("delta")) {
                (false, _) => choice,
                (true, None) => &empty,
                (true, Some(delta)) => dict(delta)?,
            };
            content.push_str(&text(delta.get(if chat { "content" } else { "text" }))?);
            for part in list(delta.get("tool_calls"))? {
                let part = dict(&part)?;
                let index = part.get("index").cloned().unwrap_or(json!(0));
                if matches!(index, Value::Array(_) | Value::Object(_)) {
                    return Err(uncaught(&index, "tool call indexes"));
                }
                let at = match calls.iter().position(|(key, _)| python_eq(key, &index)) {
                    Some(at) => at,
                    None => {
                        let call = json!({"function": {"name": "", "arguments": ""}});
                        calls.push((index, call.as_object().expect("an object").clone()));
                        calls.len() - 1
                    }
                };
                let call = &mut calls[at].1;
                let id = part
                    .get("id")
                    .filter(|id| truthy(id))
                    .or_else(|| call.get("id"))
                    .cloned()
                    .unwrap_or(Value::Null);
                call.insert("id".into(), id);
                let function = match part.get("function").filter(|value| truthy(value)) {
                    None => &empty,
                    Some(function) => dict(function)?,
                };
                for key in ["name", "arguments"] {
                    let piece = text(function.get(key))?;
                    if let Some(Value::String(joined)) = call["function"].get_mut(key) {
                        joined.push_str(&piece);
                    }
                }
            }
        }
    }
    sortable(&calls.iter().map(|(key, _)| key).collect::<Vec<_>>())?;
    calls.sort_by(|a, b| ascending(&a.0, &b.0));
    let tool_calls: Vec<Value> = calls
        .into_iter()
        .map(|(_, call)| Value::Object(call))
        .collect();
    let choice = if chat {
        json!({"finish_reason": finish,
               "message": {"content": content, "tool_calls": tool_calls}})
    } else {
        json!({"finish_reason": finish, "text": content})
    };
    let usage = if truthy(&usage) { usage } else { json!({}) };
    Ok(json!({"choices": [choice], "usage": usage}))
}

/// Validates one output, `content or ""`, and its message
/// (`structured.check_json`, `check_choice`, `check_regex`,
/// `check_tools`).
fn validate(
    validator: &Validator,
    content: &Value,
    message: &Value,
    constants: &Constants,
) -> Result<(), Fault> {
    let require = |condition: bool, message: &dyn Fn() -> String| {
        if condition {
            Ok(())
        } else {
            Err(Fault::value(message()))
        }
    };
    match validator {
        Validator::Json(schema) => {
            let value = loads(content)?;
            require(matches!(value, Object::Dict(_)), &|| {
                "output is not a JSON object".into()
            })?;
            if let Some(schema) = schema {
                let errors = schema_errors(&value, schema, "$", 0)?;
                require(errors.is_empty(), &|| joined(&errors))?;
            }
            Ok(())
        }
        Validator::Choice => {
            let offered = constants
                .choices
                .as_array()
                .is_some_and(|choices| choices.iter().any(|choice| python_eq(content, choice)));
            require(offered, &|| {
                format!("{} is not an offered choice", repr::repr(content))
            })
        }
        Validator::Regex => {
            let Value::String(text) = content else {
                return Err(Fault::type_error(format!(
                    "expected string or bytes-like object, got '{}'",
                    type_name(content)
                )));
            };
            let pattern = fancy_regex::Regex::new(&format!(r"\A(?:{})\z", constants.phone))
                .map_err(|error| Fault::Uncaught(format!("re.error: {error}")))?;
            let matched = pattern
                .is_match(text)
                .map_err(|error| Fault::Uncaught(format!("re.error: {error}")))?;
            require(matched, &|| {
                format!(
                    "{} does not fully match {}",
                    repr::repr(content),
                    constants.phone
                )
            })
        }
        Validator::Tools(named) => {
            let calls = match message.get("tool_calls").filter(|calls| truthy(calls)) {
                None => Vec::new(),
                Some(calls) => {
                    iterate(calls).map_err(|error| Fault::type_error(error.to_string()))?
                }
            };
            require(!calls.is_empty(), &|| "no tool_calls".into())?;
            for call in &calls {
                let function = item(call, "function")?;
                let name = item(function, "name")?;
                if matches!(name, Value::Array(_) | Value::Object(_)) {
                    return Err(Fault::type_error(format!(
                        "unhashable type: '{}'",
                        type_name(name)
                    )));
                }
                let schema = name
                    .as_str()
                    .and_then(|name| constants.parameters.get(name));
                require(schema.is_some(), &|| {
                    format!("tool {} was not offered", repr::repr(name))
                })?;
                require(
                    named.is_none_or(|named| name.as_str() == Some(named)),
                    &|| {
                        format!(
                            "called {}, expected {}",
                            repr::repr(name),
                            repr::repr(&json!(named))
                        )
                    },
                )?;
                let name = name.as_str().expect("an offered name");
                let arguments = loads(item(function, "arguments")?)?;
                require(matches!(arguments, Object::Dict(_)), &|| {
                    format!("{name} arguments are not a JSON object")
                })?;
                let errors = schema_errors(&arguments, schema.expect("checked"), "$", 0)?;
                require(errors.is_empty(), &|| {
                    format!("{name}: {}", joined(&errors))
                })?;
            }
            Ok(())
        }
    }
}

/// The first five schema errors joined (`"; ".join(errors[:5])`).
fn joined(errors: &[String]) -> String {
    errors[..errors.len().min(5)].join("; ")
}

/// `json.loads(value)`: a decode error is caught, a recursion error is
/// not, and anything but a string is a `TypeError`.
fn loads(value: &Value) -> Result<Object, Fault> {
    let Value::String(text) = value else {
        return Err(Fault::type_error(format!(
            "the JSON object must be str, bytes or bytearray, not {}",
            type_name(value)
        )));
    };
    pyjson::loads(text).map_err(|error| match error {
        LoadError::Decode(message) => Fault::caught("JSONDecodeError", message),
        LoadError::Value(message) => Fault::value(message),
        LoadError::Recursion(message) => Fault::Uncaught(format!("RecursionError: {message}")),
    })
}

/// Python's frames below `schema_errors` when the module recurses; past
/// its recursion limit (1000) the run stops.
const SCHEMA_DEPTH: usize = 990;

/// The items of a schema keyword as Python iterates them.
fn keyword_items(value: &Value) -> Result<Vec<Value>, Fault> {
    iterate(value).map_err(|error| Fault::type_error(error.to_string()))
}

/// Python's `TYPES[name](value)`.
fn has_type(name: &Value, value: &Object) -> Result<bool, Fault> {
    let Some(name) = name.as_str() else {
        return Err(match name {
            Value::Array(_) | Value::Object(_) => {
                Fault::type_error(format!("unhashable type: '{}'", type_name(name)))
            }
            other => Fault::caught("KeyError", repr::repr(other)),
        });
    };
    Ok(match name {
        "object" => matches!(value, Object::Dict(_)),
        "array" => matches!(value, Object::List(_)),
        "string" => matches!(value, Object::Str(_)),
        "boolean" => matches!(value, Object::Bool(_)),
        "null" => matches!(value, Object::None),
        "integer" => match value {
            Object::Int(_) | Object::BigInt(_) => true,
            Object::Float(number) => number.is_finite() && number.fract() == 0.0,
            _ => false,
        },
        "number" => matches!(value, Object::Int(_) | Object::BigInt(_) | Object::Float(_)),
        other => return Err(Fault::caught("KeyError", repr::repr(&json!(other)))),
    })
}

/// Whether `left <op> right` holds for a count or number against a schema
/// bound, as Python compares them (`TypeError` for a bound that is not a
/// number).
fn exceeds(left: &Object, op: &str, right: &Value) -> Result<bool, Fault> {
    if crate::json::python_number(right).is_none() {
        return Err(Fault::type_error(format!(
            "'{op}' not supported between instances of '{}' and '{}'",
            left.type_name(),
            type_name(right)
        )));
    }
    let wanted = if op == "<" {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    Ok(left.compare(right) == Some(wanted))
}

/// The violations of the supported JSON Schema subset, empty when valid
/// (`structured.schema_errors`). A schema that is not a dict is not one
/// the module's constants can declare; it stops the run.
fn schema_errors(
    value: &Object,
    schema: &Value,
    path: &str,
    depth: usize,
) -> Result<Vec<String>, Fault> {
    if depth > SCHEMA_DEPTH {
        return Err(Fault::Uncaught(
            "RecursionError: maximum recursion depth exceeded".into(),
        ));
    }
    let Some(schema) = schema.as_object() else {
        return Err(Fault::Uncaught(format!(
            "{path}: the schema is a {}, not a dict",
            type_name(schema)
        )));
    };
    if let Some(options) = schema.get("anyOf") {
        let mut matched = false;
        for option in keyword_items(options)? {
            if schema_errors(value, &option, path, depth + 1)?.is_empty() {
                matched = true;
                break;
            }
        }
        if !matched {
            return Ok(vec![format!("{path}: matches no anyOf branch")]);
        }
    }
    if let Some(constant) = schema.get("const")
        && (!value.equals(constant) || !value.same_type(constant))
    {
        return Ok(vec![format!(
            "{path}: expected const {}",
            repr::repr(constant)
        )]);
    }
    if let Some(options) = schema.get("enum")
        && !keyword_items(options)?
            .iter()
            .any(|item| value.equals(item) && value.same_type(item))
    {
        return Ok(vec![format!("{path}: {} not in enum", value.repr())]);
    }
    if let Some(kind) = schema.get("type").filter(|kind| !kind.is_null()) {
        let names = match kind {
            Value::String(_) => vec![kind.clone()],
            other => keyword_items(other)?,
        };
        let mut typed = false;
        for name in &names {
            if has_type(name, value)? {
                typed = true;
                break;
            }
        }
        if !typed {
            return Ok(vec![format!(
                "{path}: expected {}, got {}",
                repr::str(kind),
                value.type_name()
            )]);
        }
    }
    let mut errors = Vec::new();
    let length = |count: usize| Object::Int(count as i128);
    match value {
        Object::Dict(entries) => {
            let empty = Map::new();
            let properties = match schema.get("properties") {
                None => &empty,
                Some(Value::Object(properties)) => properties,
                Some(other) => {
                    return Err(Fault::Uncaught(format!(
                        "{path}: the properties are a {}, not a dict",
                        type_name(other)
                    )));
                }
            };
            for key in keyword_items(schema.get("required").unwrap_or(&json!([])))? {
                if matches!(key, Value::Array(_) | Value::Object(_)) {
                    return Err(Fault::type_error(format!(
                        "unhashable type: '{}'",
                        type_name(&key)
                    )));
                }
                let present = key
                    .as_str()
                    .is_some_and(|key| entries.iter().any(|(name, _)| name == key));
                if !present {
                    errors.push(format!("{path}: missing {}", repr::repr(&key)));
                }
            }
            if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                for (key, _) in entries {
                    if !properties.contains_key(key) {
                        errors.push(format!("{path}: unexpected {}", repr::repr(&json!(key))));
                    }
                }
            }
            for (key, child) in entries {
                if let Some(child_schema) = properties.get(key) {
                    errors.extend(schema_errors(
                        child,
                        child_schema,
                        &format!("{path}.{key}"),
                        depth + 1,
                    )?);
                }
            }
        }
        Object::List(items) => {
            if let Some(minimum) = schema.get("minItems")
                && exceeds(&length(items.len()), "<", minimum)?
            {
                errors.push(format!("{path}: fewer than {} items", repr::str(minimum)));
            }
            if let Some(maximum) = schema.get("maxItems")
                && exceeds(&length(items.len()), ">", maximum)?
            {
                errors.push(format!("{path}: more than {} items", repr::str(maximum)));
            }
            let empty = json!({});
            let child_schema = schema.get("items").unwrap_or(&empty);
            for (index, child) in items.iter().enumerate() {
                errors.extend(schema_errors(
                    child,
                    child_schema,
                    &format!("{path}[{index}]"),
                    depth + 1,
                )?);
            }
        }
        Object::Str(text) => {
            if let Some(maximum) = schema.get("maxLength")
                && exceeds(&length(text.chars().count()), ">", maximum)?
            {
                errors.push(format!("{path}: longer than {}", repr::str(maximum)));
            }
        }
        Object::Int(_) | Object::BigInt(_) | Object::Float(_) => {
            if let Some(minimum) = schema.get("minimum")
                && exceeds(value, "<", minimum)?
            {
                errors.push(format!("{path}: below {}", repr::str(minimum)));
            }
            if let Some(maximum) = schema.get("maximum")
                && exceeds(value, ">", maximum)?
            {
                errors.push(format!("{path}: above {}", repr::str(maximum)));
            }
        }
        Object::None | Object::Bool(_) => {}
    }
    Ok(errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speeds(values: Value) -> Result<f64, KindError> {
        let values: Vec<&Value> = values.as_array().unwrap().iter().collect();
        mean(&values, values.len())
    }

    #[test]
    fn speeds_average_as_python_sums_them() {
        // Python 3.12: sum(values) / len(values).
        for (values, expected) in [
            (
                json!([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                0.1,
            ),
            (json!([1e16, 1.0, -1e16]), 0.3333333333333333),
            (json!([3, 4]), 3.5),
            (json!([true, 2, 0.1, 0.2, 7]), 2.06),
            (json!([1, 2]), 1.5),
        ] {
            assert_eq!(
                speeds(values.clone()).unwrap().to_bits(),
                f64::to_bits(expected),
                "{values}"
            );
        }
        assert_eq!(
            speeds(json!([1, "fast"])).unwrap_err().to_string(),
            "unsupported operand type(s) for +: 'int' and 'str'"
        );
    }

    #[test]
    fn percentiles_index_the_sorted_values() {
        let values = json!([0.4, 0.1, 0.3, 0.2, 1]);
        let values: Vec<&Value> = values.as_array().unwrap().iter().collect();
        assert_eq!(percentile(&values, 0.5).unwrap(), json!(0.3));
        assert_eq!(percentile(&values, 0.9).unwrap(), json!(1));
        assert_eq!(percentile(&[], 0.5).unwrap(), Value::Null);
        let mixed = json!([1, null]);
        let mixed: Vec<&Value> = mixed.as_array().unwrap().iter().collect();
        assert!(percentile(&mixed, 0.5).is_err());
    }
}
