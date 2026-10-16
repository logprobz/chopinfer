//! `determinism.v1`: bit-identical C1 greedy answers of two server starts
//! (arms), of the two cold long-prompt sends within each, and of each
//! prefix-hit send and its cold token-ID send
//! (docs/check-kinds.md#determinism-determinismv1, `tools/qual/determinism.py`).

use std::cmp::Ordering;

use serde_json::{Map, Value, json};

use crate::capture::{self, flag, option, receipt};
use crate::declaration::Check;
use crate::json::{self, python_cmp, python_eq, python_number, sha256_hex, truthy};
use crate::kind::{
    Capture, Judged, Kind, KindError, KindInfo, MissingMetric, SHARED_KINDS, Status, finite,
    natural,
};
use crate::kinds::{applied_constants, invalid};
use crate::limits::{ANSWER_FIELDS, POSITION_FIELDS, first_difference};
use crate::repr;

pub struct Determinism;

/// The short completions, by name (`determinism.SHORT_PROMPTS`).
const SHORT_PROMPTS: [(&str, &str); 4] = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the ocean."),
    ("fibonacci", "def fibonacci(n):"),
    (
        "arithmetic",
        "Question: What is 17 multiplied by 23? Answer:",
    ),
];
const MAX_TOKENS: u64 = 48;
const TOP_LOGPROBS: u64 = 5;
/// The profiles' prefill chunk. The long prompt must need more than one.
const PREFILL_CHUNK_TOKENS: u64 = 2048;
const LONG: &str = "long";
const LONG_REPEAT: &str = "long-cold-repeat";
const IMAGE: &str = "image";
const LONG_IDS: &str = "long-ids";
/// Primer lengths in long-prompt tokens: a page and granule multiple, and
/// neither.
const HIT_ANCHORS: [u64; 2] = [1024, 2000];
const LONG_RULES: usize = 92;
const IMAGE_QUESTION: &str = "Name the four colors in this image, clockwise from the top left.";
/// `determinism.image_png`: a 256 x 256 RGB PNG of four solid quadrants,
/// its image data compressed by zlib at level 9. The receipts compare its
/// digest across arms, so these are Python's bytes.
const IMAGE_PNG: &[u8] = include_bytes!("determinism.png");

/// `determinism.SAMPLING`.
fn sampling() -> Value {
    json!({"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "top_logprobs": TOP_LOGPROBS})
}

/// The anchor of a prefix-hit request (`determinism.HITS`).
fn hit_anchor(name: &str) -> Option<u64> {
    HIT_ANCHORS
        .into_iter()
        .find(|anchor| name == format!("long-hit-{anchor}"))
}

/// The anchor of a primer request (`determinism.PRIMERS`).
fn primer_anchor(name: &str) -> Option<u64> {
    HIT_ANCHORS
        .into_iter()
        .find(|anchor| name == format!("long-primer-{anchor}"))
}

/// `determinism.hit_names`.
fn hit_names() -> Vec<String> {
    let mut names = vec![LONG_IDS.to_owned()];
    for anchor in HIT_ANCHORS {
        names.push(format!("long-primer-{anchor}"));
        names.push(format!("long-hit-{anchor}"));
    }
    names
}

/// `determinism.request_names`.
fn request_names(image: bool) -> Vec<String> {
    let mut names: Vec<String> = SHORT_PROMPTS
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .collect();
    names.extend([LONG.to_owned(), LONG_REPEAT.to_owned()]);
    if image {
        names.push(IMAGE.to_owned());
    }
    names.extend(hit_names());
    names
}

/// Python's type name of a JSON value, for its error messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python's `(value or {}).get(key)`: `None` for a falsy value, an
/// `AttributeError` for any other value that is not a dict.
fn or_get<'a>(value: Option<&'a Value>, key: &str) -> Result<Option<&'a Value>, String> {
    match value {
        None => Ok(None),
        Some(value) if !truthy(value) => Ok(None),
        Some(Value::Object(map)) => Ok(map.get(key)),
        Some(other) => Err(format!(
            "'{}' object has no attribute 'get'",
            type_name(other)
        )),
    }
}

/// The rows of `value or []` as Python's `row.get` reads them: each must be
/// a dict.
fn rows_of(value: Option<&Value>) -> Result<Vec<&Map<String, Value>>, String> {
    let no_get = |value: &Value| format!("'{}' object has no attribute 'get'", type_name(value));
    match value {
        None => Ok(Vec::new()),
        Some(value) if !truthy(value) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| item.as_object().ok_or_else(|| no_get(item)))
            .collect(),
        // Iterating a dict or a string gives strings.
        Some(Value::Object(_) | Value::String(_)) => Err(no_get(&json!(""))),
        Some(other) => Err(format!("'{}' object is not iterable", type_name(other))),
    }
}

/// `determinism.cached_tokens`.
fn cached_tokens(row: &Map<String, Value>) -> Result<Option<&Value>, String> {
    or_get(
        or_get(row.get("usage"), "prompt_tokens_details")?,
        "cached_tokens",
    )
}

/// Why a row's reported cached tokens break its rule, or `None`
/// (`determinism.cache_error`).
fn cache_error(name: &str, row: &Map<String, Value>) -> Result<Option<String>, String> {
    let cached = cached_tokens(row)?;
    let count = cached
        .filter(|cached| natural(cached))
        .and_then(Value::as_u64);
    Ok(match hit_anchor(name) {
        Some(anchor) if count.is_some_and(|count| 0 < count && count <= anchor) => None,
        Some(anchor) => Some(format!(
            "{name}: a prefix hit requires reported 0 < cached_tokens <= {anchor}, got {}",
            repr::str_of(cached)
        )),
        None if count == Some(0) => None,
        None => Some(format!(
            "{name}: a cold request requires reported cached_tokens=0, got {}",
            repr::str_of(cached)
        )),
    })
}

/// A row's cache salt when it is a string.
fn salt(row: &Map<String, Value>) -> Option<&str> {
    row.get("cache_salt").and_then(Value::as_str)
}

/// A Python int (not a bool, not a float).
fn int(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Number(number)) if !number.is_f64())
}

/// `int(value) and value > bound`.
fn int_above(value: Option<&Value>, bound: u64) -> bool {
    int(value) && python_cmp(value.expect("an int"), &json!(bound)) == Some(Ordering::Greater)
}

/// Python's `a == b` where either side may be a missing key (`None`).
fn eq(a: Option<&Value>, b: Option<&Value>) -> bool {
    python_eq(a.unwrap_or(&Value::Null), b.unwrap_or(&Value::Null))
}

/// `ValueError` unless `value` is a complete determinism record: every
/// request cold except the prefix hits, which report 0 < cached <= anchor
/// (`determinism.check`). The error is the Python exception's text.
fn complete(value: &Value) -> Result<(), String> {
    let need = |condition: bool, message: &dyn Fn() -> String| {
        if condition {
            Ok(())
        } else {
            Err(format!("determinism record: {}", message()))
        }
    };
    let map = value.as_object();
    need(
        map.is_some_and(|map| {
            map.get("passed") == Some(&Value::Bool(true))
                && !map.get("mechanics_only").is_some_and(truthy)
        }),
        &|| "not a passing model record".into(),
    )?;
    let map = map.expect("checked");
    need(
        eq(map.get("concurrency"), Some(&json!(1))) && eq(map.get("sampling"), Some(&sampling())),
        &|| "wrong workload".into(),
    )?;
    let image = or_get(map.get("image"), "included")?.and_then(Value::as_bool);
    need(image.is_some(), &|| "missing image decision".into())?;
    let names: Vec<Value> = request_names(image.expect("checked"))
        .into_iter()
        .map(Value::from)
        .collect();
    need(
        or_get(map.get("request_set"), "names")?.is_some_and(|got| python_eq(got, &json!(names)))
            && map["request_set"].get("sha256").is_some_and(truthy),
        &|| "wrong request set".into(),
    )?;
    let requests = rows_of(map.get("requests"))?;
    let got: Vec<Value> = requests
        .iter()
        .map(|row| row.get("name").cloned().unwrap_or(Value::Null))
        .collect();
    need(python_eq(&json!(got), &json!(names)), &|| {
        "missing or reordered requests".into()
    })?;
    need(
        or_get(map.get("cold_repeat"), "supported")? == Some(&Value::Bool(true)),
        &|| "missing cold long repeat".into(),
    )?;
    let hits = or_get(map.get("prefix_hit"), "supported")? == Some(&Value::Bool(true))
        && eq(
            or_get(map.get("prefix_hit"), "anchors")?,
            Some(&json!(HIT_ANCHORS)),
        );
    need(hits, &|| "missing prefix hits".into())?;
    // The names are those of the request set: distinct strings.
    let rows: Vec<(&str, &Map<String, Value>)> = requests
        .iter()
        .map(|row| (row["name"].as_str().expect("a request name"), *row))
        .collect();
    let row = |name: &str| {
        rows.iter()
            .find(|(each, _)| *each == name)
            .map(|(_, row)| *row)
            .expect("a request of the set")
    };
    let own: Vec<&str> = rows
        .iter()
        .filter(|(name, _)| hit_anchor(name).is_none())
        .filter_map(|(_, row)| salt(row))
        .collect();
    need(
        rows.iter()
            .all(|(_, row)| salt(row).is_some_and(|salt| !salt.is_empty()))
            && own
                .iter()
                .enumerate()
                .all(|(index, salt)| !own[..index].contains(salt))
            && HIT_ANCHORS.iter().all(|anchor| {
                salt(row(&format!("long-hit-{anchor}")))
                    == salt(row(&format!("long-primer-{anchor}")))
            }),
        &|| "every request needs its own cache salt, and each hit its primer's".into(),
    )?;
    for (name, row) in &rows {
        let usage = row.get("usage").filter(|usage| truthy(usage));
        need(
            matches!(
                row.get("finish_reason").and_then(Value::as_str),
                Some("stop" | "length")
            ),
            &|| format!("{name}: bad finish reason"),
        )?;
        need(row.get("text").is_some_and(Value::is_string), &|| {
            format!("{name}: missing text")
        })?;
        need(cache_error(name, row)?.is_none(), &|| {
            if hit_anchor(name).is_some() {
                format!("{name}: not a prefix hit within its anchor")
            } else {
                format!("{name}: not cold (cached tokens reported)")
            }
        })?;
        let ids = row
            .get("token_ids")
            .and_then(Value::as_array)
            .filter(|ids| !ids.is_empty() && ids.iter().all(natural));
        need(ids.is_some(), &|| {
            format!("{name}: empty or invalid token IDs")
        })?;
        let ids = ids.expect("checked").len() as u64;
        let limit = if primer_anchor(name).is_some() {
            1
        } else {
            MAX_TOKENS
        };
        need(
            eq(
                Some(&json!(ids)),
                usage.and_then(|usage| usage.get("completion_tokens")),
            ) && ids <= limit,
            &|| format!("{name}: token IDs differ from usage"),
        )?;
        let list = |key: &str| row.get(key).and_then(Value::as_array);
        let (tokens, logprobs, top) = (list("tokens"), list("logprobs"), list("top_logprobs"));
        need(
            match (tokens, logprobs, top) {
                (Some(tokens), Some(logprobs), Some(top)) => {
                    !tokens.is_empty()
                        && tokens.len() == logprobs.len()
                        && logprobs.len() == top.len()
                }
                _ => false,
            },
            &|| format!("{name}: incomplete logprobs"),
        )?;
        need(logprobs.expect("checked").iter().all(finite), &|| {
            format!("{name}: invalid selected logprobs")
        })?;
        let entry = |entry: &Value| {
            entry
                .as_array()
                .is_some_and(|pair| pair.len() == 2 && pair[0].is_string() && finite(&pair[1]))
        };
        need(
            top.expect("checked").iter().all(|entries| {
                entries
                    .as_array()
                    .is_some_and(|entries| !entries.is_empty() && entries.iter().all(entry))
            }),
            &|| format!("{name}: invalid top logprobs"),
        )?;
    }
    // The cache rules above hold each row's usage as a dict.
    let prompt_tokens = |name: &str| row(name)["usage"].get("prompt_tokens");
    let prompt = prompt_tokens(LONG);
    need(int_above(prompt, PREFILL_CHUNK_TOKENS), &|| {
        format!(
            "long prompt has {} tokens; it must exceed one {PREFILL_CHUNK_TOKENS}-token prefill chunk",
            repr::str_of(prompt)
        )
    })?;
    need(eq(prompt_tokens(LONG_REPEAT), prompt), &|| {
        "the long sends prefilled different prompts".into()
    })?;
    let request_set = &map["request_set"];
    let count = request_set.get("long_prompt_tokens");
    need(
        int_above(count, HIT_ANCHORS[1])
            && request_set
                .get("long_prompt_token_ids_sha256")
                .is_some_and(truthy),
        &|| "missing long prompt token IDs".into(),
    )?;
    let full = std::iter::once(LONG_IDS.to_owned()).chain(
        HIT_ANCHORS
            .iter()
            .map(|anchor| format!("long-hit-{anchor}")),
    );
    let mut prefilled = true;
    for name in full {
        prefilled &= eq(prompt_tokens(&name), count);
    }
    for anchor in HIT_ANCHORS {
        prefilled &= eq(
            prompt_tokens(&format!("long-primer-{anchor}")),
            Some(&json!(anchor)),
        );
    }
    need(prefilled, &|| {
        "the token-ID sends prefilled the wrong number of tokens".into()
    })
}

/// The largest |difference| of a selected or top logprob at positions where
/// both answers carry the same tokens so far, 0.0 when none
/// (`determinism.largest_difference`).
fn largest_difference(left: &Value, right: &Value) -> f64 {
    let column = |row: &Value, key: &str| row[key].as_array().cloned().unwrap_or_default();
    let (left_tokens, right_tokens) = (column(left, "tokens"), column(right, "tokens"));
    let (left_logprobs, right_logprobs) = (column(left, "logprobs"), column(right, "logprobs"));
    let (left_top, right_top) = (column(left, "top_logprobs"), column(right, "top_logprobs"));
    let float = |value: &Value| python_number(value).expect("checked finite");
    let mut largest = 0.0_f64;
    for index in 0..left_tokens.len().min(right_tokens.len()) {
        if !python_eq(&left_tokens[index], &right_tokens[index]) {
            break;
        }
        let at = |largest: f64, a: &Value, b: &Value| largest.max((float(a) - float(b)).abs());
        largest = at(largest, &left_logprobs[index], &right_logprobs[index]);
        // Python's dict of the pairs: a repeated token keeps its last value.
        let others = right_top[index].as_array().expect("checked pairs");
        for pair in left_top[index].as_array().expect("checked pairs") {
            if let Some(other) = others.iter().rev().find(|other| other[0] == pair[0]) {
                largest = at(largest, &pair[1], &other[1]);
            }
        }
    }
    largest
}

/// The first difference and largest |difference| over (name, left, right)
/// answers (`determinism.difference_report`).
fn difference_report<'a>(
    pairs: impl IntoIterator<Item = (&'a Value, &'a Value, &'a Value)>,
) -> Result<Value, KindError> {
    let (mut first, mut largest) = (Value::Null, 0.0_f64);
    for (name, left, right) in pairs {
        largest = largest.max(largest_difference(left, right));
        let (Some(a), Some(b)) = (left.as_object(), right.as_object()) else {
            return Err(invalid("an answer is not an object"));
        };
        let difference = first_difference(a, b, &POSITION_FIELDS, &ANSWER_FIELDS)
            .map_err(|error| invalid(error.0))?;
        if let Some(difference) = difference
            && first.is_null()
        {
            first = json!({"request": name, "field": difference.field,
                           "position": difference.position, "left": difference.left,
                           "right": difference.right});
        }
    }
    Ok(json!({"equal": first.is_null(), "first_difference": first, "max_abs_difference": largest}))
}

/// The request named `name` of a checked record.
fn request<'a>(value: &'a Value, name: &str) -> &'a Value {
    value["requests"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["name"] == name))
        .expect("a checked record holds every request")
}

/// The two cold long sends of a checked record (`determinism.repeat_difference`).
fn repeat_difference(value: &Value) -> Result<Value, KindError> {
    let name = json!(LONG_REPEAT);
    difference_report([(&name, request(value, LONG), request(value, LONG_REPEAT))])
}

/// Each prefix hit of a checked record against its cold token-ID send
/// (`determinism.hit_difference`).
fn hit_difference(value: &Value) -> Result<Value, KindError> {
    let names: Vec<Value> = HIT_ANCHORS
        .iter()
        .map(|anchor| json!(format!("long-hit-{anchor}")))
        .collect();
    let cold = request(value, LONG_IDS);
    difference_report(
        names
            .iter()
            .map(|name| (name, cold, request(value, name.as_str().expect("a name")))),
    )
}

/// Bit-identical answers across two records, between the cold long sends
/// within each, and between each prefix hit and its record's cold token-ID
/// send (`determinism.compare`).
fn compare(left: &Value, right: &Value) -> Result<Map<String, Value>, KindError> {
    complete(left).map_err(invalid)?;
    complete(right).map_err(invalid)?;
    if !python_eq(&left["request_set"], &right["request_set"])
        || !python_eq(&left["image"]["included"], &right["image"]["included"])
    {
        return Err(invalid("the records ran different request sets"));
    }
    let rows = |value: &'_ Value| value["requests"].as_array().cloned().unwrap_or_default();
    let (left_rows, right_rows) = (rows(left), rows(right));
    let across = difference_report(
        left_rows
            .iter()
            .zip(&right_rows)
            .map(|(a, b)| (&a["name"], a, b)),
    )?;
    let repeats = json!({"left": repeat_difference(left)?, "right": repeat_difference(right)?});
    let hits = json!({"left": hit_difference(left)?, "right": hit_difference(right)?});
    let passed = across["equal"] == true
        && ["left", "right"]
            .iter()
            .all(|side| repeats[side]["equal"] == true && hits[side]["equal"] == true);
    let tokens: usize = left_rows
        .iter()
        .map(|row| row["token_ids"].as_array().map_or(0, Vec::len))
        .sum();
    let mut metrics = Map::new();
    metrics.insert("passed".into(), json!(passed));
    metrics.insert(
        "left_arm".into(),
        left.get("arm").cloned().unwrap_or(Value::Null),
    );
    metrics.insert(
        "right_arm".into(),
        right.get("arm").cloned().unwrap_or(Value::Null),
    );
    metrics.insert("requests".into(), json!(left_rows.len()));
    metrics.insert("tokens".into(), json!(tokens));
    if let Value::Object(across) = across {
        metrics.extend(across);
    }
    metrics.insert("cold_repeat".into(), repeats);
    metrics.insert("prefix_hit".into(), hits);
    Ok(metrics)
}

// The live record (`determinism.record`).

/// One request: name, path and body without model and cache salt.
type Request = (String, &'static str, Value);

/// `determinism.COLD_REPEAT`.
fn cold_repeat() -> Value {
    json!({"supported": true, "mechanism": "cache_salt",
           "note": "each cold request carries a fresh cache_salt and must report zero cached tokens, \
                    so both long sends prefill cold within one server start"})
}

/// `determinism.PREFIX_HIT` with its anchors.
fn prefix_hit() -> Value {
    json!({"supported": true, "mechanism": "cache_salt",
           "note": "each long-hit-A reuses the salt of its A-token primer and must report \
                    0 < cached_tokens <= A; its answer must equal the cold long-ids send",
           "anchors": HIT_ANCHORS})
}

/// A completion body (`dict(determinism.COMPLETION, prompt=prompt)`).
fn completion(prompt: Value, max_tokens: u64) -> Value {
    json!({"temperature": 0, "top_p": 1, "max_tokens": max_tokens, "logprobs": TOP_LOGPROBS,
           "return_token_ids": true, "prompt": prompt})
}

/// 92 numbered rules and a question, longer than one prefill chunk
/// (`determinism.long_prompt`).
fn long_prompt() -> String {
    const COLORS: [&str; 7] = ["red", "blue", "green", "yellow", "white", "black", "orange"];
    const GOODS: [&str; 11] = [
        "apples", "bolts", "candles", "maps", "nails", "paper", "rope", "salt", "tea", "wire",
        "yarn",
    ];
    const MARKERS: [&str; 5] = ["square", "circle", "triangle", "star", "arrow"];
    const DAYS: [&str; 6] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let rules: Vec<String> = (1..=LONG_RULES)
        .map(|n| {
            format!(
                "Rule {n}: A {} box that holds {} must be placed on shelf {}, next to the {} \
                 marker, and checked every {}.",
                COLORS[n % 7],
                GOODS[n % 11],
                n % 9 + 1,
                MARKERS[n % 5],
                DAYS[n % 6]
            )
        })
        .collect();
    format!(
        "The warehouse keeps every box in order with the rules below.\n\n{}\n\nQuestion: Under \
         rule 58, what does the box hold, on which shelf is it placed, and on which day is it \
         checked? Answer:",
        rules.join("\n")
    )
}

/// Standard base64 with padding (`base64.b64encode`).
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let bits = chunk.iter().enumerate().fold(0u32, |bits, (index, byte)| {
            bits | u32::from(*byte) << (16 - 8 * index)
        });
        for index in 0..4 {
            text.push(if index <= chunk.len() {
                char::from(ALPHABET[(bits >> (18 - 6 * index) & 63) as usize])
            } else {
                '='
            });
        }
    }
    text
}

/// Every text request, then the image chat request when its chat template
/// arguments are given (`determinism.request_set`).
fn request_set(image: Option<&Value>) -> Vec<Request> {
    let mut requests: Vec<Request> = SHORT_PROMPTS
        .iter()
        .map(|(name, prompt)| {
            (
                (*name).to_owned(),
                "/v1/completions",
                completion(json!(prompt), MAX_TOKENS),
            )
        })
        .collect();
    let long = completion(json!(long_prompt()), MAX_TOKENS);
    requests.push((LONG.into(), "/v1/completions", long.clone()));
    requests.push((LONG_REPEAT.into(), "/v1/completions", long));
    if let Some(chat_template_kwargs) = image {
        let url = format!("data:image/png;base64,{}", base64(IMAGE_PNG));
        let mut body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": IMAGE_QUESTION}]}],
            "temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "logprobs": true,
            "top_logprobs": TOP_LOGPROBS, "return_token_ids": true});
        if truthy(chat_template_kwargs) {
            body["chat_template_kwargs"] = chat_template_kwargs.clone();
        }
        requests.push((IMAGE.into(), "/v1/chat/completions", body));
    }
    requests
}

/// The token-ID requests over the long prompt's IDs
/// (`determinism.hit_requests`).
fn hit_requests(token_ids: &[Value]) -> Vec<Request> {
    let mut requests: Vec<Request> = vec![(
        LONG_IDS.into(),
        "/v1/completions",
        completion(json!(token_ids), MAX_TOKENS),
    )];
    for anchor in HIT_ANCHORS {
        let primer = &token_ids[..token_ids.len().min(anchor as usize)];
        requests.push((
            format!("long-primer-{anchor}"),
            "/v1/completions",
            completion(json!(primer), 1),
        ));
        requests.push((
            format!("long-hit-{anchor}"),
            "/v1/completions",
            completion(json!(token_ids), MAX_TOKENS),
        ));
    }
    requests
}

/// `determinism.ids_digest`.
fn ids_digest(token_ids: &[Value]) -> String {
    sha256_hex(&json::compact(&json!(token_ids)))
}

/// `determinism.request_digest`: the SHA-256 of the canonical JSON of
/// every [name, path, body].
fn request_digest(requests: &[Request]) -> String {
    let rows: Vec<Value> = requests
        .iter()
        .map(|(name, path, body)| json!([name, path, body]))
        .collect();
    sha256_hex(&json::canonical(&Value::Array(rows)))
}

/// The items Python's `for item in value` visits where each item must be a
/// dict: a list's items, none of an empty string or dict; `None` for
/// anything else.
fn iterated(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::Array(items) => Some(items.clone()),
        Value::String(_) | Value::Object(_) if !truthy(value) => Some(Vec::new()),
        _ => None,
    }
}

/// The answer fields of one non-streaming response (`determinism.parse`).
fn parse(path: &str, response: &Value) -> Result<Map<String, Value>, KindError> {
    let malformed = |what: &str| invalid(format!("{path}: the response has no valid {what}"));
    let choice = response
        .get("choices")
        .and_then(|choices| choices.get(0))
        .filter(|choice| choice.is_object())
        .ok_or_else(|| malformed("choices[0]"))?;
    let some = |value: Option<&Value>| value.filter(|value| truthy(value)).cloned();
    // `for item in value or []` over items that must be dicts.
    let list = |value: Option<&Value>, what: &str| match some(value) {
        None => Ok(Vec::new()),
        Some(value) => iterated(&value).ok_or_else(|| malformed(what)),
    };
    let mut answer = Map::new();
    if path == "/v1/chat/completions" {
        let message = choice
            .get("message")
            .filter(|message| message.is_object())
            .ok_or_else(|| malformed("message"))?;
        let logprobs = some(choice.get("logprobs"));
        if logprobs
            .as_ref()
            .is_some_and(|logprobs| !logprobs.is_object())
        {
            return Err(malformed("logprobs"));
        }
        let rows = list(
            logprobs.as_ref().and_then(|rows| rows.get("content")),
            "logprobs",
        )?;
        let entry = |row: &Value, key: &str| row.get(key).cloned().ok_or_else(|| malformed(key));
        let (mut tokens, mut selected, mut top) = (Vec::new(), Vec::new(), Vec::new());
        for row in &rows {
            tokens.push(entry(row, "token")?);
            selected.push(entry(row, "logprob")?);
            let mut pairs = Vec::new();
            let items = row
                .get("top_logprobs")
                .and_then(iterated)
                .ok_or_else(|| malformed("top_logprobs"))?;
            for item in items {
                pairs.push(json!([entry(&item, "token")?, entry(&item, "logprob")?]));
            }
            top.push(Value::Array(pairs));
        }
        answer.insert(
            "text".into(),
            some(message.get("content")).unwrap_or_else(|| json!("")),
        );
        answer.insert(
            "reasoning".into(),
            some(message.get("reasoning"))
                .or_else(|| message.get("reasoning_content").cloned())
                .unwrap_or(Value::Null),
        );
        answer.insert("tokens".into(), Value::Array(tokens));
        answer.insert("logprobs".into(), Value::Array(selected));
        answer.insert("top_logprobs".into(), Value::Array(top));
    } else {
        let rows = some(choice.get("logprobs")).unwrap_or_else(|| json!({}));
        if !rows.is_object() {
            return Err(malformed("logprobs"));
        }
        let text = choice.get("text").ok_or_else(|| malformed("text"))?;
        // Kept as ordered pairs: receipts sort object keys.
        let mut top = Vec::new();
        for row in list(rows.get("top_logprobs"), "top_logprobs")? {
            let pairs = match some(Some(&row)) {
                None => Vec::new(),
                Some(Value::Object(entries)) => entries
                    .into_iter()
                    .map(|(token, logprob)| json!([token, logprob]))
                    .collect(),
                Some(_) => return Err(malformed("top_logprobs")),
            };
            top.push(Value::Array(pairs));
        }
        answer.insert("text".into(), text.clone());
        answer.insert("reasoning".into(), Value::Null);
        answer.insert(
            "tokens".into(),
            some(rows.get("tokens")).unwrap_or_else(|| json!([])),
        );
        answer.insert(
            "logprobs".into(),
            some(rows.get("token_logprobs")).unwrap_or_else(|| json!([])),
        );
        answer.insert("top_logprobs".into(), Value::Array(top));
    }
    let finish = choice
        .get("finish_reason")
        .ok_or_else(|| malformed("finish_reason"))?;
    // In the order Python's dict literal writes them.
    let mut row = Map::new();
    for key in ["text", "reasoning"] {
        row.insert(key.into(), answer[key].clone());
    }
    row.insert("finish_reason".into(), finish.clone());
    row.insert(
        "token_ids".into(),
        choice.get("token_ids").cloned().unwrap_or(Value::Null),
    );
    for key in ["tokens", "logprobs", "top_logprobs"] {
        row.insert(key.into(), answer[key].clone());
    }
    row.insert(
        "usage".into(),
        response.get("usage").cloned().unwrap_or(Value::Null),
    );
    Ok(row)
}

/// A fresh cache salt, `determinism-<name>-<uuid4 hex>`.
fn fresh_salt(name: &str) -> Result<String, KindError> {
    Ok(format!("determinism-{name}-{}", crate::capture::salt()?))
}

/// Sends one request under `salt`, or a fresh one, and reads its answer;
/// fails unless its reported cached tokens keep the row's rule
/// (`determinism.send`).
fn send(
    capture: &mut Capture<'_>,
    (name, path, body): &Request,
    salt: Option<String>,
) -> Result<Map<String, Value>, KindError> {
    let salt = match salt {
        Some(salt) => salt,
        None => fresh_salt(name)?,
    };
    let mut body = body.clone();
    body["model"] = json!(capture.server.model);
    body["cache_salt"] = json!(salt);
    capture::remaining(capture)?;
    let (response, seconds) = capture
        .client
        .json(path, Some(&body))
        .map_err(capture::http_error)?;
    let mut row = Map::new();
    row.insert("name".into(), json!(name));
    row.insert("path".into(), json!(path));
    row.insert("cache_salt".into(), json!(salt));
    row.insert("seconds".into(), json!(seconds));
    row.extend(parse(path, &response)?);
    if let Some(error) = cache_error(name, &row).map_err(invalid)? {
        return Err(invalid(error));
    }
    Ok(row)
}

/// The long prompt's token IDs from `/tokenize` (`determinism.tokenize`).
fn tokenize(capture: &mut Capture<'_>, prompt: &str) -> Result<Vec<Value>, KindError> {
    let body = json!({"model": capture.server.model, "prompt": prompt});
    capture::remaining(capture)?;
    let (response, _) = capture
        .client
        .json("/tokenize", Some(&body))
        .map_err(capture::http_error)?;
    let count = response.get("count");
    match response.get("tokens").and_then(Value::as_array) {
        Some(tokens)
            if !tokens.is_empty()
                && tokens.iter().all(natural)
                && eq(count, Some(&json!(tokens.len()))) =>
        {
            Ok(tokens.clone())
        }
        _ => Err(invalid(format!(
            "/tokenize returned no valid token IDs for the long prompt: {}",
            repr::str_of(count)
        ))),
    }
}

impl Kind for Determinism {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[3]
    }

    /// Sends the request set one request at a time (C1), the long prompt's
    /// token IDs from `/tokenize`, and the cold token-ID send, primers and
    /// prefix hits over them; the arm's record (`determinism.record`). The
    /// check's arguments name the arm (`--arm`), the run (`--run-id`), and
    /// whether the image request is sent (`--image`, `--image-reason`).
    fn capture(&self, capture: &mut Capture<'_>) -> Result<Value, KindError> {
        applied_constants(capture.check, &[])?;
        let named = |name: &str| option(capture.check, name).filter(|value| !value.is_empty());
        let (Some(arm), Some(run_id)) = (named("--arm"), named("--run-id")) else {
            return Err(invalid("record needs --arm and --run-id"));
        };
        let image = flag(capture.check, "--image");
        let reason = named("--image-reason").unwrap_or_else(|| {
            if image {
                "the server serves image inputs"
            } else {
                "the server does not serve image inputs"
            }
            .into()
        });
        let chat_template_kwargs = if image {
            Some(capture::answer_kwargs(capture)?)
        } else {
            None
        };
        let mut requests = request_set(chat_template_kwargs.as_ref());
        // One request at a time: C1.
        let mut rows = Vec::new();
        for request in &requests {
            rows.push(send(capture, request, None)?);
        }
        let prompt = long_prompt();
        let token_ids = tokenize(capture, &prompt)?;
        let hits = hit_requests(&token_ids);
        let mut salts = Map::new();
        for request in &hits {
            // A hit reuses its primer's salt, so only the primer's capture
            // can serve it.
            let salt = hit_anchor(&request.0)
                .and_then(|anchor| salts.get(&format!("long-primer-{anchor}")))
                .and_then(Value::as_str)
                .map(String::from);
            let row = send(capture, request, salt)?;
            salts.insert(request.0.clone(), row["cache_salt"].clone());
            rows.push(row);
        }
        requests.extend(hits);
        let names: Vec<&str> = requests.iter().map(|(name, _, _)| name.as_str()).collect();
        let mut value = json!({
            "arm": arm, "run_id": run_id, "concurrency": 1, "sampling": sampling(),
            "request_set": {"names": names, "sha256": request_digest(&requests),
                            "long_prompt_sha256": sha256_hex(prompt.as_bytes()),
                            "long_prompt_tokens": token_ids.len(),
                            "long_prompt_token_ids_sha256": ids_digest(&token_ids)},
            "image": {"included": image, "png_sha256": image.then(|| sha256_hex(IMAGE_PNG)),
                      "reason": reason},
            "cold_repeat": cold_repeat(), "prefix_hit": prefix_hit(),
            "requests": rows, "passed": true});
        complete(&value).map_err(invalid)?;
        let differences = [
            ("cold_repeat", repeat_difference(&value)?),
            ("prefix_hit", hit_difference(&value)?),
        ];
        for (key, difference) in differences {
            if let (Some(entry), Value::Object(difference)) =
                (value[key].as_object_mut(), difference)
            {
                entry.extend(difference);
            }
        }
        let Value::Object(result) = value else {
            unreachable!("a record is an object")
        };
        Ok(receipt(&format!("determinism-{arm}"), capture, result))
    }

    /// Both arms are complete determinism records (`kinds.Determinism.verify`).
    /// The module's constants are fixed: a check that sets one is refused.
    fn verify(
        &self,
        value: &Value,
        reference: Option<&Value>,
        check: &Check,
    ) -> Result<Vec<String>, KindError> {
        applied_constants(check, &[])?;
        let reference = reference
            .filter(|reference| !reference.is_null())
            .ok_or_else(|| KindError::Insufficient("no record of the other arm".into()))?;
        for record in [value, reference] {
            complete(record).map_err(KindError::Insufficient)?;
        }
        Ok(Vec::new())
    }

    /// `determinism.compare` of this arm's record and the other arm's
    /// (`reference`).
    fn score(
        &self,
        value: &Value,
        reference: Option<&Value>,
        _check: &Check,
        _mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        compare(value, reference.unwrap_or(&Value::Null))
    }

    /// Pass when every compared answer is bit-equal; no predicates.
    fn judge(
        &self,
        metrics: &Map<String, Value>,
        _check: &Check,
    ) -> Result<(Status, Vec<Judged>), MissingMetric> {
        let passed = metrics
            .get("passed")
            .ok_or_else(|| MissingMetric("'passed'".into()))?;
        let status = if truthy(passed) {
            Status::Pass
        } else {
            Status::Fail
        };
        Ok((status, Vec::new()))
    }
}
