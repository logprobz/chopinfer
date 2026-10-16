//! A model's qualification declaration, `models/<m>/qual.json`
//! (`chopinfer-qual-model/v1`), validated here only (`tools/qual/
//! declaration.py` asks `eval declaration`): every object has a fixed set of
//! fields, and a missing, unknown or mistyped field is refused with its path.
//! [`expand`] states each of its configurations as a declaration of its own.
//!
//! The modules a check may name and the type each constant override must
//! have come from `tools/qual/modules.json` under the same root, which
//! `tools/qual/module_metadata.py` derives from the module sources; an
//! install ships it without Python.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::json::{self, is_digest};

pub const SCHEMA: &str = "chopinfer-qual-model/v1";
pub const FILE_NAME: &str = "qual.json";
pub const OPERATORS: [&str; 5] = ["eq", "le", "ge", "count_ge", "true_count_ge"];
pub const CHANGE_CLASSES: [&str; 5] = ["frontend", "quality", "service", "speculation", "prefix"];
/// The checkpoint pin fields a reference's provenance can be held to.
pub const CHECKPOINT_KEYS: [&str; 5] = [
    "repository",
    "revision",
    "served_model_name",
    "source_lock_sha256",
    "checkpoint_identity",
];
const CHECK_FIELDS: [&str; 15] = [
    "module",
    "args",
    "timeout_seconds",
    "receipt",
    "stdout_json",
    "json_file",
    "reference_file",
    "reference_equal",
    "rules",
    "files",
    "constants",
    "input_rules",
    "cohort_args",
    "informational_reason",
    "waivers",
];
/// The metrics a category waiver may name: a teacher category's spread.
pub const WAIVER_METRICS: [&str; 4] = [
    "mean_abs_dlogprob",
    "top1_decisive",
    "top1_near_tie",
    "top1_disagreement",
];
const TOP_REQUIRED: [&str; 8] = [
    "schema",
    "model",
    "cohorts",
    "change_classes",
    "checks",
    "inputs",
    "qualification",
    "receipts",
];
const TOP_OPTIONAL: [&str; 9] = [
    "configurations",
    "references",
    "checkpoint",
    "limitations",
    "preview",
    "renamed_checks",
    "capabilities",
    "variant",
    "kv_offload",
];
/// The module metadata, relative to the root.
pub const MODULES: &str = "tools/qual/modules.json";
pub const MODULES_SCHEMA: &str = "chopinfer-qual-modules/v1";

/// The qualification modules a check may name, each with its top-level
/// names and the JSON type an override of each must have.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Modules(pub BTreeMap<String, BTreeMap<String, String>>);

impl Modules {
    /// Reads `tools/qual/modules.json` under `root`.
    pub fn read(root: &Path) -> Checked<Self> {
        let path = root.join(MODULES);
        let (_, value) = json::read(&path).map_err(DeclarationError)?;
        let invalid = || {
            DeclarationError(format!(
                "{}: not {MODULES_SCHEMA} module metadata",
                path.display()
            ))
        };
        if value["schema"].as_str() != Some(MODULES_SCHEMA) {
            return Err(invalid());
        }
        let mut modules = BTreeMap::new();
        for (module, constants) in value["modules"].as_object().ok_or_else(invalid)? {
            let mut kinds = BTreeMap::new();
            for (name, kind) in constants.as_object().ok_or_else(invalid)? {
                kinds.insert(name.clone(), kind.as_str().ok_or_else(invalid)?.to_owned());
            }
            modules.insert(module.clone(), kinds);
        }
        Ok(Self(modules))
    }
}

/// Refuses an override `value` of a constant of `kind`
/// (`declaration.check_constant`).
fn constant(value: &Value, field: &str, kind: &str) -> Checked<()> {
    let (valid, wanted) = match kind {
        "number" => (finite(value).is_some(), "must be a finite number"),
        "int" => (
            matches!(value, Value::Number(number) if !number.is_f64()),
            "must be an integer",
        ),
        "bool" => (value.is_boolean(), "must be true or false"),
        "str" => (value.is_string(), "must be a string"),
        "list" => (value.is_array(), "must be a list"),
        "dict" => (value.is_object(), "must be a dict"),
        "null" => (value.is_null(), "must be a NoneType"),
        _ => (false, "has a type no JSON value has"),
    };
    if valid { Ok(()) } else { fail(field, wanted) }
}

/// A declaration field is missing or malformed; the message names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclarationError(pub String);

impl std::fmt::Display for DeclarationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DeclarationError {}

type Checked<T> = Result<T, DeclarationError>;

fn fail<T>(field: &str, message: &str) -> Checked<T> {
    Err(DeclarationError(format!("{field}: {message}")))
}

fn object<'a>(value: &'a Value, field: &str) -> Checked<&'a Map<String, Value>> {
    match value.as_object() {
        Some(map) => Ok(map),
        None => fail(field, "must be an object"),
    }
}

/// An object with every required field and no field outside both lists.
fn fields<'a>(
    value: &'a Value,
    field: &str,
    required: &[&str],
    optional: &[&str],
) -> Checked<&'a Map<String, Value>> {
    let map = object(value, field)?;
    for key in required {
        if !map.contains_key(*key) {
            return fail(&format!("{field}.{key}"), "is required");
        }
    }
    for key in map.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return fail(&format!("{field}.{key}"), "unknown field");
        }
    }
    Ok(map)
}

fn strings<'a>(value: &'a Value, field: &str, nonempty: bool) -> Checked<Vec<&'a str>> {
    let items: Option<Vec<&str>> = value
        .as_array()
        .and_then(|items| items.iter().map(Value::as_str).collect());
    match items {
        None => fail(field, "must be a list of strings"),
        Some(items) if nonempty && items.is_empty() => fail(field, "must not be empty"),
        Some(items) => Ok(items),
    }
}

fn text<'a>(value: &'a Value, field: &str) -> Checked<&'a str> {
    match value.as_str() {
        Some(text) if !text.trim().is_empty() => Ok(text),
        _ => fail(field, "must be a nonempty string"),
    }
}

fn choice<'a>(value: &'a Value, field: &str, choices: &[&str]) -> Checked<&'a str> {
    match value.as_str() {
        Some(text) if choices.contains(&text) => Ok(text),
        _ => fail(field, &format!("must be one of {}", choices.join(", "))),
    }
}

fn boolean(value: &Value, field: &str) -> Checked<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        _ => fail(field, "must be true or false"),
    }
}

/// A finite number that is not a bool: Python's `is_number`.
fn finite(value: &Value) -> Option<f64> {
    value.as_f64().filter(|value| value.is_finite())
}

fn positive(value: &Value, field: &str) -> Checked<f64> {
    match finite(value) {
        Some(number) if number > 0.0 => Ok(number),
        _ => fail(field, "must be a positive number"),
    }
}

/// A JSON integer (not a float or bool), as Python's `type(v) is int`.
fn integer(value: &Value) -> Option<i128> {
    match value {
        Value::Number(number) if !number.is_f64() => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from)),
        _ => None,
    }
}

fn positive_integer(value: &Value, field: &str) -> Checked<u64> {
    match integer(value) {
        Some(number) if number > 0 => Ok(number as u64),
        _ => fail(field, "must be a positive integer"),
    }
}

fn positive_integers(value: &Value, field: &str) -> Checked<Vec<u64>> {
    match value.as_array() {
        Some(items) if !items.is_empty() => items
            .iter()
            .enumerate()
            .map(|(index, item)| positive_integer(item, &format!("{field}[{index}]")))
            .collect(),
        _ => fail(field, "must be a nonempty list of positive integers"),
    }
}

fn digest<'a>(value: &'a Value, field: &str) -> Checked<&'a str> {
    match value.as_str() {
        Some(text) if is_digest(text) => Ok(text),
        _ => fail(field, "must be a lowercase sha256 hex digest"),
    }
}

/// A declared predicate: `{path, op, value, source}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Predicate {
    pub path: Vec<Value>,
    pub op: String,
    pub value: Value,
    pub source: String,
}

pub(crate) fn rules(value: &Value, field: &str) -> Checked<Vec<Predicate>> {
    let items = match value.as_array() {
        Some(items) if !items.is_empty() => items,
        _ => return fail(field, "must be a nonempty list of predicates"),
    };
    let mut result = Vec::new();
    for (index, rule) in items.iter().enumerate() {
        let where_ = format!("{field}[{index}]");
        let map = fields(rule, &where_, &["path", "op", "value", "source"], &[])?;
        let path = match map["path"].as_array() {
            Some(keys)
                if keys
                    .iter()
                    .all(|key| key.is_string() || key.is_boolean() || integer(key).is_some()) =>
            {
                keys.clone()
            }
            _ => return fail(&format!("{where_}.path"), "must be a list of keys"),
        };
        let op = choice(&map["op"], &format!("{where_}.op"), &OPERATORS)?;
        if matches!(op, "le" | "ge") && finite(&map["value"]).is_none() {
            return fail(
                &format!("{where_}.value"),
                &format!("must be a finite number for {op}"),
            );
        }
        if matches!(op, "count_ge" | "true_count_ge")
            && integer(&map["value"]).is_none_or(|v| v <= 0)
        {
            return fail(
                &format!("{where_}.value"),
                &format!("must be a positive integer for {op}"),
            );
        }
        let source = text(&map["source"], &format!("{where_}.source"))?;
        result.push(Predicate {
            path,
            op: op.into(),
            value: map["value"].clone(),
            source: source.into(),
        });
    }
    Ok(result)
}

/// A declared input: a path relative to the root, or a path under an
/// environment variable's directory with an optional default directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    Unused,
    Path(String),
    Environment {
        env: String,
        default: Option<String>,
        path: String,
    },
}

fn input(value: &Value, field: &str) -> Checked<Input> {
    match value {
        Value::Null => Ok(Input::Unused),
        Value::String(path) => Ok(Input::Path(path.clone())),
        _ => {
            let map = fields(value, field, &["env"], &["default", "path"])?;
            let env = text(&map["env"], &format!("{field}.env"))?;
            let read = |key: &str| match map.get(key) {
                None => Ok(None),
                Some(Value::String(text)) => Ok(Some(text.clone())),
                Some(_) => fail(&format!("{field}.{key}"), "must be a string"),
            };
            Ok(Input::Environment {
                env: env.into(),
                default: read("default")?,
                path: read("path")?.unwrap_or_default(),
            })
        }
    }
}

/// One declared check.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub id: String,
    pub module: String,
    pub timeout_seconds: f64,
    pub rules: Vec<Predicate>,
    pub informational_reason: Option<String>,
    /// The check as declared, for what a kind reads beyond these fields.
    pub declared: Value,
}

fn check(
    id: &str,
    value: &Value,
    cohorts: &Map<String, Value>,
    modules: &Modules,
) -> Checked<Check> {
    let field = format!("checks.{id}");
    let map = fields(
        value,
        &field,
        &["module", "args", "timeout_seconds", "rules"],
        &CHECK_FIELDS,
    )?;
    let module = match map["module"].as_str() {
        Some(name) if modules.0.contains_key(name) => name,
        _ => {
            return fail(
                &format!("{field}.module"),
                "must name an existing qualification module",
            );
        }
    };
    strings(&map["args"], &format!("{field}.args"), false)?;
    let timeout_seconds = positive(&map["timeout_seconds"], &format!("{field}.timeout_seconds"))?;
    if !["receipt", "stdout_json", "json_file"]
        .iter()
        .any(|key| map.contains_key(*key))
    {
        return fail(
            &format!("{field}.receipt"),
            "is required unless stdout_json or json_file names the receipt",
        );
    }
    for key in ["receipt", "json_file", "reference_file"] {
        if let Some(value) = map.get(key) {
            text(value, &format!("{field}.{key}"))?;
        }
    }
    if map.contains_key("reference_file") && !map.contains_key("json_file") {
        return fail(
            &format!("{field}.reference_file"),
            "needs json_file: the kind scores that file against this one",
        );
    }
    if let Some(paths) = map.get("reference_equal") {
        if !map.contains_key("reference_file") {
            return fail(
                &format!("{field}.reference_equal"),
                "needs reference_file: it compares that record with the receipt",
            );
        }
        let Some(paths) = paths.as_array().filter(|paths| !paths.is_empty()) else {
            return fail(
                &format!("{field}.reference_equal"),
                "must be a nonempty list of paths",
            );
        };
        for (index, path) in paths.iter().enumerate() {
            strings(path, &format!("{field}.reference_equal.{index}"), true)?;
        }
    }
    if map
        .get("stdout_json")
        .is_some_and(|value| value != &Value::Bool(true))
    {
        return fail(&format!("{field}.stdout_json"), "must be true when present");
    }
    let predicates = rules(&map["rules"], &format!("{field}.rules"))?;
    if let Some(files) = map.get("files") {
        strings(files, &format!("{field}.files"), false)?;
    }
    let empty = Value::Object(Map::new());
    let constants = object(
        map.get("constants").unwrap_or(&empty),
        &format!("{field}.constants"),
    )?;
    for (module, entries) in constants {
        let where_ = format!("{field}.constants.{module}");
        let Some(defaults) = modules.0.get(module) else {
            return fail(&where_, "must name an existing qualification module");
        };
        for (name, entry) in object(entries, &where_)? {
            let at = format!("{where_}.{name}");
            let Some(kind) = defaults.get(name) else {
                return fail(&at, &format!("is not a top-level name of {module}.py"));
            };
            let entry = fields(entry, &at, &["value", "source"], &[])?;
            constant(&entry["value"], &format!("{at}.value"), kind)?;
            text(&entry["source"], &format!("{at}.source"))?;
        }
    }
    let input_rules = object(
        map.get("input_rules").unwrap_or(&empty),
        &format!("{field}.input_rules"),
    )?;
    for (key, value) in input_rules {
        rules(value, &format!("{field}.input_rules.{key}"))?;
    }
    let cohort_args = object(
        map.get("cohort_args").unwrap_or(&empty),
        &format!("{field}.cohort_args"),
    )?;
    for (cohort, arguments) in cohort_args {
        if !cohorts.contains_key(cohort) {
            return fail(
                &format!("{field}.cohort_args.{cohort}"),
                "names no declared cohort",
            );
        }
        strings(arguments, &format!("{field}.cohort_args.{cohort}"), false)?;
    }
    let informational_reason = match map.get("informational_reason") {
        Some(value) => Some(text(value, &format!("{field}.informational_reason"))?.to_owned()),
        None => None,
    };
    if let Some(value) = map.get("waivers") {
        waivers(value, &format!("{field}.waivers"), module, &predicates)?;
    }
    Ok(Check {
        id: id.into(),
        module: module.into(),
        timeout_seconds,
        rules: predicates,
        informational_reason,
        declared: value.clone(),
    })
}

/// A check's declared waivers: each a recorded decision the judge applies
/// to one failing predicate, re-judged against the waiver's `ceiling` in
/// place of the rule's value (`predicate`: the rule's path and op; the
/// ceiling must be looser than the rule), or to one teacher category whose
/// only failing metric is `metric`, at most `ceiling` (`category`, for a
/// check that declares `passed` true). Each names its `reason` and its
/// `decision` (who decided, when, where it is recorded).
fn waivers(value: &Value, field: &str, module: &str, rules: &[Predicate]) -> Checked<()> {
    let Some(items) = value.as_array().filter(|items| !items.is_empty()) else {
        return fail(field, "must be a nonempty list of waivers");
    };
    for (index, item) in items.iter().enumerate() {
        let at = format!("{field}.{index}");
        let map = fields(
            item,
            &at,
            &["ceiling", "reason", "decision"],
            &["predicate", "category", "metric"],
        )?;
        text(&map["reason"], &format!("{at}.reason"))?;
        text(&map["decision"], &format!("{at}.decision"))?;
        let ceiling = &map["ceiling"];
        match (map.get("predicate"), map.get("category")) {
            (Some(predicate), None) => {
                if map.contains_key("metric") {
                    return fail(&format!("{at}.metric"), "belongs to a category waiver");
                }
                let named = fields(predicate, &format!("{at}.predicate"), &["path", "op"], &[])?;
                let Some(rule) = rules.iter().find(|rule| {
                    named["path"] == Value::Array(rule.path.clone())
                        && named["op"] == rule.op.as_str()
                }) else {
                    return fail(
                        &format!("{at}.predicate"),
                        "names no declared rule of the check",
                    );
                };
                let looser = match rule.op.as_str() {
                    "eq" => ceiling != &rule.value,
                    op => match (finite(ceiling), finite(&rule.value)) {
                        (Some(ceiling), Some(value)) if op == "le" => ceiling > value,
                        (Some(ceiling), Some(value)) => ceiling < value,
                        _ => false,
                    },
                };
                if !looser {
                    return fail(
                        &format!("{at}.ceiling"),
                        "must be a value the rule refuses: looser than its bound",
                    );
                }
            }
            (None, Some(category)) => {
                text(category, &format!("{at}.category"))?;
                let metric = map.get("metric").and_then(Value::as_str);
                if !metric.is_some_and(|metric| WAIVER_METRICS.contains(&metric)) {
                    return fail(
                        &format!("{at}.metric"),
                        &format!("must be one of {}", WAIVER_METRICS.join(", ")),
                    );
                }
                positive(ceiling, &format!("{at}.ceiling"))?;
                if !crate::kind::SHARED_KINDS[0].modules.contains(&module) {
                    return fail(
                        &format!("{at}.category"),
                        "only a teacher check has categories",
                    );
                }
                if !rules.iter().any(|rule| {
                    rule.path == [Value::from("passed")]
                        && rule.op == "eq"
                        && rule.value == Value::Bool(true)
                }) {
                    return fail(
                        &format!("{at}.category"),
                        "needs the check to declare passed eq true, which the waiver reports on",
                    );
                }
            }
            _ => return fail(&at, "must name either a predicate or a category"),
        }
    }
    Ok(())
}

/// A check's waivers in its JSON form (`waivers`, when it declares any),
/// checked by the declaration's rules against its module and rules: the one
/// waiver validator. The declaration loader, `eval rescore` (every check it
/// builds, and its `waivers` call for the Python kinds) and the judge
/// ([`crate::kind::waive`]) all apply it.
pub fn validate_waivers(check: &Value, module: &str, rules: &[Predicate]) -> Checked<()> {
    match check.get("waivers") {
        None => Ok(()),
        Some(value) => waivers(value, "waivers", module, rules),
    }
}

/// A pinned reference (`references.<name>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferencePin {
    pub name: String,
    pub path: Input,
    pub sha256: String,
    pub schema: String,
    pub variants: Option<Vec<String>>,
    pub workload_sha256: Option<String>,
    pub capture_sha256: Option<String>,
    /// The checkpoint pin fields its provenance is held to; every one when
    /// the declaration names none.
    pub checkpoint_keys: Option<Vec<String>>,
    /// The runner workload a bound reference scored: a repo path, or a path
    /// under an environment variable's directory like an input.
    pub workload: Option<Input>,
}

fn reference(name: &str, value: &Value, read_by_module: bool) -> Checked<ReferencePin> {
    let field = format!("references.{name}");
    let map = fields(
        value,
        &field,
        &["path", "sha256", "schema"],
        &[
            "variants",
            "workload_sha256",
            "capture_sha256",
            "checkpoint_keys",
            "workload",
        ],
    )?;
    let path = match &map["path"] {
        value @ Value::Object(_) => input(value, &format!("{field}.path"))?,
        value => Input::Path(text(value, &format!("{field}.path"))?.to_owned()),
    };
    let sha256 = digest(&map["sha256"], &format!("{field}.sha256"))?;
    let schema = text(&map["schema"], &format!("{field}.schema"))?;
    if read_by_module && !map.contains_key("variants") {
        return fail(
            &format!("{field}.variants"),
            "is required for a reference a check module reads",
        );
    }
    let variants = match map.get("variants") {
        Some(value) => {
            let variants = strings(value, &format!("{field}.variants"), true)?;
            let mut unique = variants.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != variants.len() {
                return fail(&format!("{field}.variants"), "must not repeat a variant");
            }
            Some(variants.into_iter().map(String::from).collect())
        }
        None => None,
    };
    let workload = match map.get("workload") {
        Some(value @ Value::Object(_)) => Some(input(value, &format!("{field}.workload"))?),
        Some(value) => Some(Input::Path(
            text(value, &format!("{field}.workload"))?.to_owned(),
        )),
        None => None,
    };
    let mut digests = [None, None];
    for (slot, key) in digests
        .iter_mut()
        .zip(["workload_sha256", "capture_sha256"])
    {
        if let Some(value) = map.get(key) {
            *slot = Some(digest(value, &format!("{field}.{key}"))?.to_owned());
        }
    }
    let [workload_sha256, capture_sha256] = digests;
    let checkpoint_keys = match map.get("checkpoint_keys") {
        Some(value) => {
            let keys = strings(value, &format!("{field}.checkpoint_keys"), true)?;
            for key in &keys {
                if !CHECKPOINT_KEYS.contains(key) {
                    return fail(
                        &format!("{field}.checkpoint_keys"),
                        &format!("must be one of {}", CHECKPOINT_KEYS.join(", ")),
                    );
                }
            }
            Some(keys.into_iter().map(String::from).collect())
        }
        None => None,
    };
    Ok(ReferencePin {
        name: name.into(),
        path,
        sha256: sha256.into(),
        schema: schema.into(),
        variants,
        workload_sha256,
        capture_sha256,
        checkpoint_keys,
        workload,
    })
}

/// The checkpoint a gate is pinned to (`qualification.checkpoint`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointPin {
    /// The repo path of the checkpoint's source lock.
    pub lock: String,
    pub repository: String,
    pub revision: String,
    pub served_model_name: String,
    pub source_lock_sha256: String,
    pub checkpoint_identity: String,
    /// The pinned entry of a checkpoint registry: (`pin`, `registry`).
    pub registry: Option<(String, String)>,
}

impl CheckpointPin {
    /// The pin's value of one of [`CHECKPOINT_KEYS`].
    pub fn key(&self, key: &str) -> Option<&str> {
        Some(match key {
            "repository" => &self.repository,
            "revision" => &self.revision,
            "served_model_name" => &self.served_model_name,
            "source_lock_sha256" => &self.source_lock_sha256,
            "checkpoint_identity" => &self.checkpoint_identity,
            _ => return None,
        })
    }
}

fn qualification(value: &Value) -> Checked<CheckpointPin> {
    let field = "qualification";
    if value
        .as_object()
        .is_some_and(|map| map.contains_key("references"))
    {
        return fail(
            "qualification.references",
            "declare references at the top level",
        );
    }
    let map = fields(
        value,
        field,
        &[
            "teacher_reference_kind",
            "teacher_receipt",
            "teacher_capture_schema",
            "verify_buckets",
            "chars_per_token",
            "checkpoint",
        ],
        &["prefix", "needle", "mechanics_only", "media"],
    )?;
    choice(
        &map["teacher_reference_kind"],
        "qualification.teacher_reference_kind",
        &["exact", "prompt-logprobs"],
    )?;
    text(&map["teacher_receipt"], "qualification.teacher_receipt")?;
    text(
        &map["teacher_capture_schema"],
        "qualification.teacher_capture_schema",
    )?;
    positive_integers(&map["verify_buckets"], "qualification.verify_buckets")?;
    positive(&map["chars_per_token"], "qualification.chars_per_token")?;
    if let Some(value) = map.get("mechanics_only") {
        boolean(value, "qualification.mechanics_only")?;
    }
    if let Some(media) = map.get("media") {
        let at = "qualification.media";
        let media = object(media, at)?;
        let format = choice(
            media.get("format").unwrap_or(&Value::Null),
            &format!("{at}.format"),
            &["cases", "image-probes"],
        )?;
        let extra = if format == "cases" {
            vec!["capture_schema", "expectations", "audio_near_tie"]
        } else {
            vec!["stop_token", "expected", "fixture_images", "items"]
        };
        let mut required = vec!["format", "reference", "receipt"];
        required.extend(extra);
        // Image probes may name the thinking switch their requests send.
        let optional: &[&str] = if format == "image-probes" {
            &["audio_near_tie", "coverage", "thinking"]
        } else {
            &["audio_near_tie", "coverage"]
        };
        let media = fields(&map["media"], at, &required, optional)?;
        if let Some(thinking) = media.get("thinking") {
            boolean(thinking, &format!("{at}.thinking"))?;
        }
        for key in required.into_iter().filter(|key| *key != "format") {
            let field = format!("{at}.{key}");
            if ["expected", "fixture_images", "items", "expectations"].contains(&key) {
                object(&media[key], &field)?;
            } else if key != "audio_near_tie" {
                text(&media[key], &field)?;
            }
        }
        if let Some(coverage) = media.get("coverage") {
            for (kind, items) in object(coverage, &format!("{at}.coverage"))? {
                choice(
                    &Value::String(kind.clone()),
                    &format!("{at}.coverage"),
                    &["image", "audio", "video"],
                )?;
                let field = format!("{at}.coverage.{kind}");
                let items = strings(items, &field, true)?;
                let mut unique = items.clone();
                unique.sort_unstable();
                unique.dedup();
                if items.iter().any(|item| item.trim().is_empty()) || items.len() != unique.len() {
                    return fail(&field, "must name unique nonempty reference items");
                }
            }
        }
        if let Some(limit) = media.get("audio_near_tie") {
            positive(limit, &format!("{at}.audio_near_tie"))?;
        }
        if format == "cases" {
            let at = format!("{at}.expectations");
            let expectations = fields(&media["expectations"], &at, &[], &["files", "kinds"])?;
            for group in ["files", "kinds"] {
                if let Some(entries) = expectations.get(group) {
                    for (name, entry) in object(entries, &format!("{at}.{group}"))? {
                        let field = format!("{at}.{group}.{name}");
                        let entry = fields(entry, &field, &["field", "single"], &[])?;
                        text(&entry["field"], &format!("{field}.field"))?;
                        boolean(&entry["single"], &format!("{field}.single"))?;
                    }
                }
            }
        }
        if format == "image-probes" {
            for (name, value) in object(&media["items"], &format!("{at}.items"))? {
                text(&Value::String(name.clone()), &format!("{at}.items"))?;
                choice(
                    value,
                    &format!("{at}.items.{name}"),
                    &["short", "free-text"],
                )?;
            }
            for (name, value) in object(&media["expected"], &format!("{at}.expected"))? {
                if !media["items"].as_object().unwrap().contains_key(name) {
                    return fail(
                        &format!("{at}.expected.{name}"),
                        "must name a declared item",
                    );
                }
                strings(value, &format!("{at}.expected.{name}"), true)?;
            }
            for (name, value) in object(&media["fixture_images"], &format!("{at}.fixture_images"))?
            {
                if !media["items"].as_object().unwrap().contains_key(name) {
                    return fail(
                        &format!("{at}.fixture_images.{name}"),
                        "must name a declared item",
                    );
                }
                text(value, &format!("{at}.fixture_images.{name}"))?;
            }
        }
    }
    let mut required = vec!["lock"];
    required.extend(CHECKPOINT_KEYS);
    let pin = fields(
        &map["checkpoint"],
        "qualification.checkpoint",
        &required,
        &["pin", "registry"],
    )?;
    for key in [
        "lock",
        "repository",
        "revision",
        "served_model_name",
        "pin",
        "registry",
    ] {
        if let Some(value) = pin.get(key) {
            text(value, &format!("qualification.checkpoint.{key}"))?;
        }
    }
    for key in ["source_lock_sha256", "checkpoint_identity"] {
        digest(&pin[key], &format!("qualification.checkpoint.{key}"))?;
    }
    let registry = match (pin.get("pin"), pin.get("registry")) {
        (Some(name), Some(registry)) => Some((
            name.as_str().unwrap_or_default().to_owned(),
            registry.as_str().unwrap_or_default().to_owned(),
        )),
        (None, None) => None,
        (Some(_), None) => {
            return fail("qualification.checkpoint.registry", "is required with pin");
        }
        (None, Some(_)) => {
            return fail("qualification.checkpoint.pin", "is required with registry");
        }
    };
    if let Some(prefix) = map.get("prefix") {
        let where_ = "qualification.prefix";
        let flags = ["prompt_end", "media", "acceptance"];
        let mut required = vec!["family"];
        required.extend(flags);
        required.extend([
            "block_tokens",
            "prefill_granule",
            "anchor_boundary",
            "anchor_labels",
            "reference_schema",
        ]);
        let prefix = fields(prefix, where_, &required, &[])?;
        text(&prefix["family"], &format!("{where_}.family"))?;
        for key in flags {
            boolean(&prefix[key], &format!("{where_}.{key}"))?;
        }
        positive_integer(&prefix["block_tokens"], &format!("{where_}.block_tokens"))?;
        positive_integer(
            &prefix["prefill_granule"],
            &format!("{where_}.prefill_granule"),
        )?;
        positive_integer(
            &prefix["anchor_boundary"],
            &format!("{where_}.anchor_boundary"),
        )?;
        strings(
            &prefix["anchor_labels"],
            &format!("{where_}.anchor_labels"),
            true,
        )?;
        text(
            &prefix["reference_schema"],
            &format!("{where_}.reference_schema"),
        )?;
    }
    if let Some(needle) = map.get("needle") {
        let where_ = "qualification.needle";
        let needle = fields(needle, where_, &["format", "lengths", "match"], &["frozen"])?;
        let format = choice(
            &needle["format"],
            &format!("{where_}.format"),
            &["context-files", "prompt-set"],
        )?;
        positive_integers(&needle["lengths"], &format!("{where_}.lengths"))?;
        choice(
            &needle["match"],
            &format!("{where_}.match"),
            &["whitespace", "alphanumeric"],
        )?;
        if format == "prompt-set" && !needle.contains_key("frozen") {
            return fail(
                &format!("{where_}.frozen"),
                "is required for format prompt-set",
            );
        }
        if let Some(frozen) = needle.get("frozen") {
            text(frozen, &format!("{where_}.frozen"))?;
        }
    }
    let read = |key: &str| pin[key].as_str().unwrap_or_default().to_owned();
    Ok(CheckpointPin {
        lock: read("lock"),
        repository: read("repository"),
        revision: read("revision"),
        served_model_name: read("served_model_name"),
        source_lock_sha256: read("source_lock_sha256"),
        checkpoint_identity: read("checkpoint_identity"),
        registry,
    })
}

/// The serving and numerical facts the correctness runner reads from the
/// `preview` section (tools/qual/release_preview.py model_facts).
fn preview(value: &Value) -> Checked<()> {
    let field = "preview";
    let map = fields(
        value,
        field,
        &[
            "profile",
            "speculation",
            "spec_greedy",
            "numerical_reference",
            "gsm8k_floor",
        ],
        &["soak_prefix_mix", "package_max_sequences", "quick_arms"],
    )?;
    text(&map["profile"], "preview.profile")?;
    let speculation = fields(
        &map["speculation"],
        "preview.speculation",
        &["drafter", "tokens", "draft_cost"],
        &[],
    )?;
    text(&speculation["drafter"], "preview.speculation.drafter")?;
    positive_integer(&speculation["tokens"], "preview.speculation.tokens")?;
    choice(
        &speculation["draft_cost"],
        "preview.speculation.draft_cost",
        &["per-token", "per-block"],
    )?;
    if let Some(value) = map.get("package_max_sequences") {
        // The most sequences the model's package serves.
        positive_integer(value, "preview.package_max_sequences")?;
    }
    let greedy = fields(
        &map["spec_greedy"],
        "preview.spec_greedy",
        &["rule"],
        &["reason"],
    )?;
    let rule = choice(
        &greedy["rule"],
        "preview.spec_greedy.rule",
        &["exact", "informational"],
    )?;
    if rule == "informational" {
        match greedy.get("reason") {
            None => {
                return fail(
                    "preview.spec_greedy.reason",
                    "is required for rule informational",
                );
            }
            Some(reason) => {
                text(reason, "preview.spec_greedy.reason")?;
            }
        }
    } else if greedy.contains_key("reason") {
        return fail(
            "preview.spec_greedy.reason",
            "applies only to rule informational",
        );
    }
    text(&map["numerical_reference"], "preview.numerical_reference")?;
    if positive_integer(&map["gsm8k_floor"], "preview.gsm8k_floor")? > 200 {
        return fail(
            "preview.gsm8k_floor",
            "must be at most the screen's 200 items",
        );
    }
    if let Some(value) = map.get("soak_prefix_mix") {
        boolean(value, "preview.soak_prefix_mix")?;
    }
    if let Some(value) = map.get("quick_arms") {
        // The arms whose servers the quick cohort admits (default both).
        let arms = strings(value, "preview.quick_arms", true)?;
        for (index, arm) in arms.iter().enumerate() {
            if !["ar", "spec"].contains(arm) || arms[..index].contains(arm) {
                return fail(
                    "preview.quick_arms",
                    "must name ar and spec at most once each",
                );
            }
        }
    }
    Ok(())
}

/// A validated model declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct Declaration {
    pub model: String,
    /// The checkpoint variant a `qual.VARIANT.json` qualifies, if any.
    pub variant: Option<String>,
    /// The cohorts in declared order: name and check IDs.
    pub cohorts: Vec<(String, Vec<String>)>,
    /// The checks in declared order.
    pub checks: Vec<Check>,
    pub references: Vec<ReferencePin>,
    pub inputs: Vec<(String, Input)>,
    pub receipts: Input,
    pub checkpoint: CheckpointPin,
    pub mechanics_only: bool,
    /// The `qualification` block as declared, for the facts a kind reads
    /// beyond the checkpoint pin.
    pub qualification: Value,
    pub limitations: Vec<String>,
    /// Former check IDs and the IDs that replaced them.
    pub renamed_checks: Vec<(String, String)>,
}

impl Declaration {
    pub fn check(&self, id: &str) -> Option<&Check> {
        self.checks.iter().find(|check| check.id == id)
    }

    pub fn reference(&self, name: &str) -> Option<&ReferencePin> {
        self.references.iter().find(|pin| pin.name == name)
    }

    pub fn cohort(&self, name: &str) -> Option<&[String]> {
        self.cohorts
            .iter()
            .find(|(cohort, _)| cohort == name)
            .map(|(_, checks)| checks.as_slice())
    }

    pub fn input(&self, name: &str) -> Option<&Input> {
        self.inputs
            .iter()
            .find(|(input, _)| input == name)
            .map(|(_, input)| input)
    }
}

fn coverage_contract(check: &Value, module: &str, args: &[&str]) -> bool {
    let (command, flags, required) = match module {
        "spec_verify" => (
            "compare",
            vec!["--ar", "--spec"],
            serde_json::json!([
                [["passed"], "eq", true],
                [["sets", "copy", "scored_positions"], "ge", 1]
            ]),
        ),
        "verify_exact" => (
            "report",
            vec!["--ar", "--verify"],
            serde_json::json!([
                [["records"], "count_ge", 1],
                [["records", "*", "k"], "ge", 1],
                [["records", "*", "coverage", "passed"], "eq", true],
                [
                    ["records", "*", "exact", "all", "gate_against_ar", "passed"],
                    "eq",
                    true
                ]
            ]),
        ),
        "prefix_exact" => (
            "report",
            vec!["--record"],
            serde_json::json!([[["passed"], "eq", true]]),
        ),
        "preempt_compare" => (
            "compare",
            vec!["--reserved", "--preempted"],
            serde_json::json!([
                [["passed"], "eq", true],
                [["exact"], "eq", true],
                [["requests"], "ge", 1],
                [["positions_after_first_preemption"], "ge", 1]
            ]),
        ),
        _ => return true,
    };
    if args.first() != Some(&command) {
        return false;
    }
    let mut operands = Vec::new();
    for flag in flags {
        let values: Vec<_> = args
            .windows(2)
            .filter(|pair| pair[0] == flag)
            .map(|pair| pair[1])
            .collect();
        if values.is_empty() || values.iter().any(|v| v.starts_with("--")) {
            return false;
        }
        operands.push(values);
    }
    if operands.len() == 2 && operands[0].iter().any(|value| operands[1].contains(value)) {
        return false;
    }
    required.as_array().unwrap().iter().all(|required| {
        check["rules"].as_array().unwrap().iter().any(|rule| {
            rule["path"] == required[0]
                && rule["op"] == required[1]
                && (if required[2] == true {
                    rule["value"] == true
                } else {
                    finite(&rule["value"]).is_some_and(|v| v >= required[2].as_f64().unwrap())
                })
        })
    })
}

/// Feature coverage uses shared modules, so check IDs remain model-local.
fn capabilities(value: &Value) -> Checked<()> {
    let Some(declared) = value.get("capabilities") else {
        return Ok(());
    };
    let obligations = [
        ("text", "teacher"),
        ("frontend", "frontend_goldens"),
        ("api", "api"),
        ("structured", "structured"),
        ("prefix_cache", "prefix_exact"),
        ("preemption", "preempt_compare"),
        ("long_context", "needle"),
        ("stability", "soak"),
    ];
    let mut required: Vec<_> = obligations.iter().map(|(name, _)| *name).collect();
    required.extend(["media", "speculation", "kv_offload"]);
    let declared = fields(declared, "capabilities", &required, &["unqualified_media"])?;
    let mut needed = Vec::new();
    for (name, module) in obligations {
        if boolean(&declared[name], &format!("capabilities.{name}"))? {
            needed.push((name, module));
        }
    }
    let media = strings(&declared["media"], "capabilities.media", false)?;
    for modality in &media {
        if !["image", "audio", "video"].contains(modality) {
            return fail("capabilities.media", "must be one of image, audio, video");
        }
    }
    let mut unique = media.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != media.len() {
        return fail("capabilities.media", "must not repeat a modality");
    }
    // A modality the model serves without in-tree reference evidence is
    // declared here with its reason.
    if let Some(unqualified) = declared.get("unqualified_media") {
        let coverage = &value["qualification"]["media"]["coverage"];
        for (modality, reason) in object(unqualified, "capabilities.unqualified_media")? {
            let field = format!("capabilities.unqualified_media.{modality}");
            if !["image", "audio", "video"].contains(&modality.as_str()) {
                return fail(&field, "must be one of image, audio, video");
            }
            if media.contains(&modality.as_str()) {
                return fail(&field, "is also declared qualified in capabilities.media");
            }
            if coverage.get(modality).is_some() {
                return fail(
                    &field,
                    "must not keep reference items in qualification.media.coverage",
                );
            }
            text(reason, &field)?;
        }
    }
    if !media.is_empty() {
        let policy = &value["qualification"]["media"];
        let supported: Vec<&str> = if policy["format"] == "image-probes" {
            if policy["items"]
                .as_object()
                .is_some_and(|items| !items.is_empty())
            {
                vec!["image"]
            } else {
                vec![]
            }
        } else {
            policy["coverage"]
                .as_object()
                .into_iter()
                .flat_map(|items| items.keys().map(String::as_str))
                .collect()
        };
        if media.iter().any(|kind| !supported.contains(kind)) {
            return fail(
                "capabilities.media",
                "each modality requires its own nonempty reference items",
            );
        }
        needed.push(("media", "multimodal"));
    }
    if choice(
        &declared["speculation"],
        "capabilities.speculation",
        &["none", "lookup", "mtp", "dflash"],
    )? != "none"
    {
        needed.extend([
            ("speculation", "spec_verify"),
            ("speculation", "verify_exact"),
        ]);
    }
    if let Some(policy) = value.get("kv_offload") {
        crate::kv_offload::facts(policy).map_err(DeclarationError)?;
    }
    if boolean(&declared["kv_offload"], "capabilities.kv_offload")? {
        if value.get("kv_offload").is_none() {
            return fail("capabilities.kv_offload", "requires kv_offload facts");
        }
        let full = value["cohorts"]["full"]["checks"].as_array();
        for scenario in ["plain", "pressure"] {
            let off = format!("kv_offload_{scenario}_off");
            let swap = format!("kv_offload_{scenario}_swap");
            let covered = full.is_some_and(|ids| {
                ids.iter().any(|id| {
                    let check = &value["checks"][id.as_str().unwrap_or_default()];
                    let args = check["args"].as_array();
                    check["module"] == "kv_offload"
                        && check.get("informational_reason").is_none()
                        && check["files"] == serde_json::json!([off, swap])
                        && args.is_some_and(|args| {
                            args == &vec![
                                serde_json::json!("report"),
                                serde_json::json!("--off"),
                                serde_json::json!(format!("{{{off}}}")),
                                serde_json::json!("--swap"),
                                serde_json::json!(format!("{{{swap}}}")),
                            ]
                        })
                        && check["rules"].as_array().is_some_and(|rules| {
                            rules.iter().any(|rule| {
                                rule["path"] == serde_json::json!(["passed"])
                                    && rule["op"] == "eq"
                                    && rule["value"] == true
                            })
                        })
                })
            });
            if !covered {
                return fail(
                    "capabilities.kv_offload",
                    &format!("requires a hard kv_offload {scenario} check in cohorts.full"),
                );
            }
        }
    }
    let full = value["cohorts"]["full"]["checks"].as_array();
    for (name, module) in needed {
        let covered = full.is_some_and(|full| {
            full.iter().any(|id| {
                let check = &value["checks"][id.as_str().unwrap_or_default()];
                if check["module"] != module || check.get("informational_reason").is_some() {
                    return false;
                }
                let args: Vec<_> = check["args"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .chain(
                        check["cohort_args"]["full"]
                            .as_array()
                            .into_iter()
                            .flatten(),
                    )
                    .filter_map(Value::as_str)
                    .collect();
                if !coverage_contract(check, module, &args) {
                    return false;
                }
                if module == "preempt_compare"
                    && (!args.contains(&"--exact")
                        || !args.contains(&"compare")
                        || args.contains(&"--baseline"))
                {
                    return false;
                }
                if module == "soak" {
                    if args.contains(&"--remote-only")
                        || !args.contains(&"--duration")
                        || !args.contains(&"--concurrency")
                    {
                        return false;
                    }
                    for (enabled, flag) in [
                        (declared["prefix_cache"] == true, "--prefix-mix"),
                        (!media.is_empty(), "--media-mix"),
                        (declared["structured"] == true, "--structured-mix"),
                    ] {
                        if enabled && !args.contains(&flag) {
                            return false;
                        }
                    }
                }
                true
            })
        });
        if !covered {
            return fail(
                &format!("capabilities.{name}"),
                &format!("requires a hard {module} check in cohorts.full"),
            );
        }
    }
    Ok(())
}

/// Refuses a malformed declaration with a message naming the field, in the
/// order `declaration.validate` checks them.
pub fn validate(value: &Value, modules: &Modules) -> Checked<Declaration> {
    let map = object(value, "declaration")?;
    if map.get("schema").and_then(Value::as_str) != Some(SCHEMA) {
        return fail("schema", &format!("must be {SCHEMA}"));
    }
    for key in map.keys() {
        if !TOP_REQUIRED.contains(&key.as_str()) && !TOP_OPTIONAL.contains(&key.as_str()) {
            return fail(key, "unknown field");
        }
    }
    for key in TOP_REQUIRED {
        if !map.contains_key(key) {
            return fail(key, "is required");
        }
    }
    let model = text(&map["model"], "model")?;
    if !plain(model) {
        return fail("model", "must be a plain name");
    }
    let variant = match map.get("variant") {
        Some(value) => {
            let variant = text(value, "variant")?;
            if !plain(variant) {
                return fail("variant", "must be a plain name");
            }
            Some(variant.to_owned())
        }
        None => None,
    };
    let checks = object(&map["checks"], "checks")?;
    if checks.is_empty() {
        return fail("checks", "must declare at least one check");
    }
    let cohorts_map = object(&map["cohorts"], "cohorts")?;
    if cohorts_map.is_empty() {
        return fail("cohorts", "must declare at least one cohort");
    }
    let mut cohorts = Vec::new();
    for (name, policy) in cohorts_map {
        let where_ = format!("cohorts.{name}");
        let policy = fields(policy, &where_, &["checks"], &[])?;
        let names = strings(&policy["checks"], &format!("{where_}.checks"), true)?;
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != names.len() {
            return fail(&format!("{where_}.checks"), "must not repeat a check");
        }
        for name in &names {
            if !checks.contains_key(*name) {
                return fail(
                    &format!("{where_}.checks"),
                    &format!("names undeclared check {name}"),
                );
            }
        }
        cohorts.push((name.clone(), names.into_iter().map(String::from).collect()));
    }
    let classes = fields(
        &map["change_classes"],
        "change_classes",
        &CHANGE_CLASSES,
        &[],
    )?;
    for (change, names) in classes {
        for name in strings(names, &format!("change_classes.{change}"), false)? {
            if !checks.contains_key(name) {
                return fail(
                    &format!("change_classes.{change}"),
                    &format!("names undeclared check {name}"),
                );
            }
        }
    }
    let mut declared_checks = Vec::new();
    for (id, value) in checks {
        declared_checks.push(check(id, value, cohorts_map, modules)?);
    }
    let empty = Value::Object(Map::new());
    let mut renamed_checks = Vec::new();
    for (old, new) in object(
        map.get("renamed_checks").unwrap_or(&empty),
        "renamed_checks",
    )? {
        if checks.contains_key(old) {
            return fail(&format!("renamed_checks.{old}"), "is a declared check");
        }
        let new = text(new, &format!("renamed_checks.{old}"))?;
        if !checks.contains_key(new) {
            return fail(
                &format!("renamed_checks.{old}"),
                &format!("names undeclared check {new}"),
            );
        }
        renamed_checks.push((old.clone(), new.to_owned()));
    }
    let references_map = object(map.get("references").unwrap_or(&empty), "references")?;
    let media_reference = map["qualification"]
        .get("media")
        .and_then(|media| media.get("reference"))
        .and_then(Value::as_str)
        .unwrap_or("media");
    let mut references = Vec::new();
    for (name, value) in references_map {
        let read_by_module = name == "teacher"
            || (name == media_reference
                && map["qualification"]["media"]["format"]
                    .as_str()
                    .unwrap_or("cases")
                    == "cases");
        references.push(reference(name, value, read_by_module)?);
    }
    for check in &declared_checks {
        if let Some(needed) = match check.module.as_str() {
            "teacher" => Some("teacher"),
            "multimodal" => Some(media_reference),
            _ => None,
        } && !references_map.contains_key(needed)
        {
            return fail(
                &format!("references.{needed}"),
                &format!("is required by checks.{}", check.id),
            );
        }
    }
    let mut inputs = Vec::new();
    for (name, value) in object(&map["inputs"], "inputs")? {
        inputs.push((name.clone(), input(value, &format!("inputs.{name}"))?));
    }
    let receipts = input(&map["receipts"], "receipts")?;
    let checkpoint = qualification(&map["qualification"])?;
    let mechanics_only = map["qualification"]
        .get("mechanics_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(status) = map.get("checkpoint") {
        for (key, value) in fields(
            status,
            "checkpoint",
            &[],
            &["repository", "revision", "status"],
        )? {
            text(value, &format!("checkpoint.{key}"))?;
        }
    }
    let limitations = strings(
        map.get("limitations").unwrap_or(&Value::Array(Vec::new())),
        "limitations",
        false,
    )?
    .into_iter()
    .map(String::from)
    .collect();
    if let Some(value) = map.get("preview") {
        preview(value)?;
    }
    capabilities(value)?;
    if let Some(declared) = map.get("configurations") {
        if variant.is_some() {
            return fail("configurations", "a variant declares no configurations");
        }
        for name in object(declared, "configurations")?.keys() {
            if name == "default" || !plain(name) {
                return fail(
                    &format!("configurations.{name}"),
                    "must be a plain name other than default",
                );
            }
            let expanded = expand(value, name)?;
            validate(&expanded, modules)
                .map_err(|error| DeclarationError(format!("configurations.{name}: {}", error.0)))?;
        }
    }
    Ok(Declaration {
        model: model.into(),
        variant,
        cohorts,
        checks: declared_checks,
        references,
        inputs,
        receipts,
        checkpoint,
        mechanics_only,
        qualification: map["qualification"].clone(),
        limitations,
        renamed_checks,
    })
}

/// Whether `name` is a plain model or variant name: letters, digits, `-`
/// and `_`, with at least one letter or digit.
fn plain(name: &str) -> bool {
    name.chars()
        .filter(|c| *c != '_' && *c != '-')
        .all(char::is_alphanumeric)
        && !name.chars().all(|c| c == '_' || c == '-')
}

/// A declaration read from its file, with the file's digest.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub declaration: Declaration,
    /// The declaration as written.
    pub value: Value,
    pub path: PathBuf,
    pub sha256: String,
    /// The directory every declared repo path resolves under: the parent of
    /// `models/` in a checkout or an install.
    pub root: PathBuf,
}

/// Reads and validates the declaration of the model directory `model`
/// (`<root>/models/<m>`, holding `qual.json`). The declaration must name
/// its directory.
pub fn load(model: &Path) -> Checked<Loaded> {
    load_variant(model, None)
}

/// The variants of the model directory `model`: the configurations its
/// `qual.json` declares and each `qual.VARIANT.json` beside it, in name
/// order.
pub fn variants(model: &Path) -> Checked<Vec<String>> {
    let entries = std::fs::read_dir(model)
        .map_err(|error| DeclarationError(format!("{}: {error}", model.display())))?;
    let mut variants = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| DeclarationError(error.to_string()))?;
        if let Some(variant) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.strip_prefix("qual."))
            .and_then(|name| name.strip_suffix(".json"))
            .filter(|variant| !variant.is_empty())
        {
            variants.push(variant.to_owned());
        }
    }
    if let Ok((_, value)) = json::read(&model.join(FILE_NAME))
        && let Some(declared) = value.get("configurations").and_then(Value::as_object)
    {
        for name in declared.keys() {
            if !variants.contains(name) {
                variants.push(name.clone());
            }
        }
    }
    variants.sort();
    Ok(variants)
}

/// As [`load`], or the declaration of `variant`: the configuration of that
/// name `qual.json` declares ([`expand`]), else `qual.VARIANT.json`, which
/// must name that variant; `qual.json` names none (`declaration.load`).
pub fn load_variant(model: &Path, variant: Option<&str>) -> Checked<Loaded> {
    let mut path = match variant {
        Some(variant) if !plain(variant) => {
            return Err(DeclarationError(format!(
                "variant {variant:?} is not a plain name"
            )));
        }
        Some(variant) => model.join(format!("qual.{variant}.json")),
        None => model.join(FILE_NAME),
    };
    let mut read = None;
    if let Some(variant) = variant {
        let base = model.join(FILE_NAME);
        let (bytes, value) = json::read(&base).map_err(DeclarationError)?;
        if value["configurations"].get(variant).is_some() {
            let expanded = expand(&value, variant)
                .map_err(|error| DeclarationError(format!("{}: {error}", base.display())))?;
            read = Some((bytes, expanded));
            path = base;
        }
    }
    let (bytes, value) = match read {
        Some(read) => read,
        None => json::read(&path).map_err(DeclarationError)?,
    };
    let directory = std::fs::canonicalize(model)
        .map_err(|error| DeclarationError(format!("{}: {error}", model.display())))?;
    let root = directory
        .parent()
        .filter(|parent| parent.file_name().is_some_and(|name| name == "models"))
        .and_then(Path::parent)
        .ok_or_else(|| {
            DeclarationError(format!(
                "{}: a model directory is ROOT/models/<model>",
                directory.display()
            ))
        })?
        .to_path_buf();
    let modules = Modules::read(&root)?;
    let declaration = validate(&value, &modules)
        .map_err(|error| DeclarationError(format!("{}: {error}", path.display())))?;
    if declaration.variant.as_deref() != variant {
        return Err(DeclarationError(format!(
            "{}: variant: {}",
            path.display(),
            match variant {
                Some(variant) => format!("must be {variant} in qual.{variant}.json"),
                None => "is not allowed in qual.json".into(),
            }
        )));
    }
    let named = directory.file_name().and_then(|name| name.to_str());
    if named != Some(declaration.model.as_str()) {
        return Err(DeclarationError(format!(
            "{}: model: must name its directory {}",
            path.display(),
            named.unwrap_or_default()
        )));
    }
    Ok(Loaded {
        declaration,
        value,
        path,
        sha256: json::sha256_hex(&bytes),
        root,
    })
}

/// The fields of a configuration (`configurations.NAME`). The runner reads
/// the first group (how the configuration is served); [`expand`] applies the
/// rest to the base declaration.
const CONFIGURATION_FIELDS: [&str; 24] = [
    "experimental",
    "checkpoint",
    "install_features",
    "tensor_parallel_size",
    "serve",
    "environment",
    "producer",
    "screens",
    "checks",
    "quick",
    "serving",
    "source",
    "media",
    "from_default",
    "references",
    "keep_references",
    "inputs",
    "qualification",
    "rules",
    "cohort_args",
    "append",
    "change_classes",
    "capabilities",
    "waivers",
];
/// The served profile facts a configuration's `serving` sets on the checks'
/// rules.
const SERVING_FIELDS: [&str; 6] = [
    "context_parallel_size",
    "kv_cache_dtype",
    "tensor_parallel_size",
    "speculation_tokens",
    "checkpoint_identity",
    "profile_name",
];
const PROFILE: [&str; 4] = ["server", "health", "serving", "profile"];

/// Whether `path` (a rule's) is `PROFILE` followed by `tail`.
fn profile_path(path: &Value, tail: &[&str]) -> bool {
    path.as_array().is_some_and(|items| {
        items.len() == PROFILE.len() + tail.len()
            && items
                .iter()
                .zip(PROFILE.iter().chain(tail))
                .all(|(item, part)| item == part)
    })
}

/// The configuration `name` of a declaration: the base with that
/// configuration's checks and cohorts, its serving facts set on every
/// profile rule (a declared input another configuration records keeps its
/// rules), its references, rule values, cohort arguments, appended rules,
/// inputs, capabilities and qualification facts, and `variant` naming it.
/// Every `reference_sha256` in `value` that names a replaced pin, repinned.
fn repin(value: &mut Value, replaced: &[(Value, Value)]) {
    match value {
        Value::Object(map) => {
            for (key, item) in map.iter_mut() {
                match replaced.iter().find(|(old, _)| old == item) {
                    Some((_, new)) if key == "reference_sha256" => *item = new.clone(),
                    _ => repin(item, replaced),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|item| repin(item, replaced)),
        _ => {}
    }
}

pub fn expand(base: &Value, name: &str) -> Checked<Value> {
    let field = format!("configurations.{name}");
    let Some(config) = base.get("configurations").and_then(|all| all.get(name)) else {
        return fail(&field, "is not declared");
    };
    let spec = object(config, &field)?;
    for key in spec.keys() {
        if !CONFIGURATION_FIELDS.contains(&key.as_str()) {
            return fail(&format!("{field}.{key}"), "unknown field");
        }
    }
    let base_checks = object(&base["checks"], "checks")?;
    let ids = strings(&spec["checks"], &format!("{field}.checks"), true)?;
    let quick = strings(&spec["quick"], &format!("{field}.quick"), false)?;
    let fixed = match spec.get("from_default") {
        Some(value) => strings(value, &format!("{field}.from_default"), false)?,
        None => Vec::new(),
    };
    let empty = Map::new();
    let serving = match spec.get("serving") {
        Some(value) => object(value, &format!("{field}.serving"))?,
        None => &empty,
    };
    for key in serving.keys() {
        if !SERVING_FIELDS.contains(&key.as_str()) {
            return fail(&format!("{field}.serving.{key}"), "unknown field");
        }
    }
    let media = match spec.get("media") {
        Some(value) => boolean(value, &format!("{field}.media"))?,
        None => true,
    };
    let source = match spec.get("source") {
        Some(value) => text(value, &format!("{field}.source"))?,
        None => "",
    };
    // Digests the configuration replaces wherever a rule pins them.
    let mut replaced: Vec<(Value, Value)> = Vec::new();
    if let Some(identity) = serving.get("checkpoint_identity") {
        replaced.push((
            base["qualification"]["checkpoint"]["checkpoint_identity"].clone(),
            identity.clone(),
        ));
    }
    let references = match spec.get("references") {
        Some(value) => object(value, &format!("{field}.references"))?.clone(),
        None => Map::new(),
    };
    for (reference, pin) in &references {
        replaced.push((
            base["references"][reference]["sha256"].clone(),
            pin["sha256"].clone(),
        ));
    }
    let section = |key: &str| -> Checked<&Map<String, Value>> {
        match spec.get(key) {
            Some(value) => object(value, &format!("{field}.{key}")),
            None => Ok(&empty),
        }
    };
    let (overrides, cohort_args, appended, waived) = (
        section("rules")?,
        section("cohort_args")?,
        section("append")?,
        section("waivers")?,
    );
    for id in waived.keys() {
        if !ids.contains(&id.as_str()) {
            return fail(
                &format!("{field}.waivers.{id}"),
                "names a check the configuration does not run",
            );
        }
    }
    let mut checks = Map::new();
    for id in &ids {
        let Some(check) = base_checks.get(*id) else {
            return fail(
                &format!("{field}.checks"),
                &format!("names undeclared check {id}"),
            );
        };
        let mut check = check.clone();
        let mut lists: Vec<&mut Vec<Value>> = Vec::new();
        let map = object(&check, &format!("checks.{id}"))?.clone();
        let mut map = map;
        map.entry("rules")
            .or_insert_with(|| Value::Array(Vec::new()));
        {
            let (rules, inputs): (Vec<_>, Vec<_>) =
                map.iter_mut().partition(|(key, _)| *key == "rules");
            for (_, value) in rules {
                if let Value::Array(items) = value {
                    lists.push(items);
                }
            }
            for (key, value) in inputs {
                if key == "input_rules"
                    && let Value::Object(inputs) = value
                {
                    for (input, value) in inputs.iter_mut() {
                        if !fixed.contains(&input.as_str())
                            && let Value::Array(items) = value
                        {
                            lists.push(items);
                        }
                    }
                }
            }
            for rules in lists {
                if !media {
                    rules.retain(|rule| rule["path"] != serde_json::json!(["image", "included"]));
                }
                for rule in rules.iter_mut() {
                    for key in [
                        "context_parallel_size",
                        "kv_cache_dtype",
                        "tensor_parallel_size",
                    ] {
                        if let Some(value) = serving.get(key)
                            && profile_path(&rule["path"], &[key])
                        {
                            rule["value"] = value.clone();
                        }
                    }
                    if let Some(tokens) = serving.get("speculation_tokens")
                        && profile_path(&rule["path"], &["speculation", "tokens"])
                        && rule["op"] == "ge"
                    {
                        rule["op"] = "eq".into();
                        rule["value"] = tokens.clone();
                    }
                    if rule["value"].is_string()
                        && let Some((_, new)) =
                            replaced.iter().find(|(old, _)| *old == rule["value"])
                    {
                        rule["value"] = new.clone();
                    }
                }
                if let Some(profile) = serving.get("profile_name") {
                    let mut named = false;
                    for rule in rules.iter_mut() {
                        if profile_path(&rule["path"], &["name"]) {
                            rule["value"] = profile.clone();
                            named = true;
                        }
                    }
                    let last = rules.iter().rposition(|rule| {
                        rule["path"].as_array().is_some_and(|path| {
                            path.len() >= 4 && path[..4] == PROFILE.map(Value::from)
                        })
                    });
                    if let (false, Some(at)) = (named, last) {
                        rules.insert(
                            at + 1,
                            serde_json::json!({"path": [PROFILE[0], PROFILE[1], PROFILE[2], PROFILE[3], "name"],
                                               "op": "eq", "value": profile, "source": source}),
                        );
                    }
                }
            }
        }
        // A check's constants name the reference they apply to by its digest
        // (paired_teacher's margins): a replaced pin repins them too.
        if let Some(constants) = map.get_mut("constants") {
            repin(constants, &replaced);
        }
        if let Some(values) = overrides.get(*id) {
            let values = values
                .as_array()
                .ok_or_else(|| DeclarationError(format!("{field}.rules.{id}: must be a list")))?;
            if let Some(Value::Array(rules)) = map.get_mut("rules") {
                for value in values {
                    for rule in rules
                        .iter_mut()
                        .filter(|rule| rule["path"] == value["path"])
                    {
                        rule["value"] = value["value"].clone();
                    }
                }
            }
        }
        match cohort_args.get(*id) {
            Some(Value::Null) => {
                map.remove("cohort_args");
            }
            Some(value) => {
                map.insert("cohort_args".into(), value.clone());
            }
            None => {}
        }
        if let Some(append) = appended.get(*id) {
            for (key, rules) in object(append, &format!("{field}.append.{id}"))? {
                let rules = rules.as_array().ok_or_else(|| {
                    DeclarationError(format!("{field}.append.{id}.{key}: must be a list"))
                })?;
                let target = if key == "rules" {
                    map.entry("rules")
                        .or_insert_with(|| Value::Array(Vec::new()))
                } else {
                    map.entry("input_rules")
                        .or_insert_with(|| Value::Object(Map::new()))
                        .as_object_mut()
                        .ok_or_else(|| {
                            DeclarationError(format!("checks.{id}.input_rules: must be an object"))
                        })?
                        .entry(key.clone())
                        .or_insert_with(|| Value::Array(Vec::new()))
                };
                let Value::Array(target) = target else {
                    return fail(&format!("checks.{id}.{key}"), "must be a list");
                };
                for rule in rules {
                    let mut rule = object(rule, &format!("{field}.append.{id}.{key}"))?.clone();
                    let unless = rule
                        .remove("unless_present")
                        .is_some_and(|value| value == true);
                    if unless && target.iter().any(|old| old["path"] == rule["path"]) {
                        continue;
                    }
                    target.push(Value::Object(rule));
                }
            }
        }
        // The base check's waivers are the default configuration's; a
        // configuration declares its own.
        map.remove("waivers");
        if let Some(list) = waived.get(*id) {
            map.insert("waivers".into(), list.clone());
        }
        check = Value::Object(map);
        checks.insert((*id).to_owned(), check);
    }
    let mut value = base.clone();
    let out = value.as_object_mut().expect("the base is an object");
    out.remove("configurations");
    let cohorts = object(&base["cohorts"], "cohorts")?;
    let cohort = |name: &str, list: &[&str]| -> Value {
        let mut cohort = cohorts
            .get(name)
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        cohort["checks"] = serde_json::json!(list);
        cohort
    };
    out.insert(
        "cohorts".into(),
        serde_json::json!({"quick": cohort("quick", &quick), "full": cohort("full", &ids)}),
    );
    let extra = section("change_classes")?;
    let mut classes = Map::new();
    for (change, names) in object(&base["change_classes"], "change_classes")? {
        let mut kept: Vec<Value> = names
            .as_array()
            .into_iter()
            .flatten()
            .filter(|check| check.as_str().is_some_and(|id| checks.contains_key(id)))
            .cloned()
            .collect();
        for check in extra
            .get(change)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if check.as_str().is_some_and(|id| checks.contains_key(id)) {
                kept.push(check.clone());
            }
        }
        classes.insert(change.clone(), Value::Array(kept));
    }
    out.insert("change_classes".into(), Value::Object(classes));
    // A former check ID stays only while the check that replaced it runs.
    if let Some(renamed) = base.get("renamed_checks").and_then(Value::as_object) {
        let kept: Map<String, Value> = renamed
            .iter()
            .filter(|(_, new)| new.as_str().is_some_and(|id| checks.contains_key(id)))
            .map(|(old, new)| (old.clone(), new.clone()))
            .collect();
        out.insert("renamed_checks".into(), Value::Object(kept));
    }
    out.insert("checks".into(), Value::Object(checks));
    let base_references = base
        .get("references")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut kept_references = Map::new();
    match spec.get("keep_references") {
        Some(names) => {
            for reference in strings(names, &format!("{field}.keep_references"), false)? {
                if let Some(pin) = base_references.get(reference) {
                    kept_references.insert(reference.to_owned(), pin.clone());
                }
            }
        }
        None => kept_references = base_references,
    }
    kept_references.extend(references);
    out.insert("references".into(), Value::Object(kept_references));
    let mut inputs = object(&base["inputs"], "inputs")?.clone();
    inputs.extend(section("inputs")?.clone());
    out.insert("inputs".into(), Value::Object(inputs));
    match spec.get("capabilities") {
        Some(Value::Null) => {
            out.remove("capabilities");
        }
        Some(capabilities) => {
            out.insert("capabilities".into(), capabilities.clone());
        }
        None => {}
    }
    let mut qualification = object(&base["qualification"], "qualification")?.clone();
    qualification.extend(section("qualification")?.clone());
    out.insert("qualification".into(), Value::Object(qualification));
    out.insert("variant".into(), Value::String(name.to_owned()));
    Ok(value)
}

/// Each declared configuration's name, after `default`.
pub fn configurations(value: &Value) -> Vec<String> {
    let mut names = vec!["default".to_owned()];
    if let Some(declared) = value.get("configurations").and_then(Value::as_object) {
        names.extend(declared.keys().cloned());
    }
    names
}
