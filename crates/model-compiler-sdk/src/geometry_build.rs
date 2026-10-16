//! The build-script half of a contract's geometry catalog.
//!
//! A contract's `build.rs` reads `geometries/supported.json` beside it, plus
//! the directories an environment variable lists, checks every declared
//! geometry document, and writes Rust declarations and values for them. The
//! model supplies its field and group tables, its structure check and the
//! items it renders; this module supplies the directory walk, rule
//! narrowing, field reading and checking, struct rendering, runtime fields,
//! aliases and tuning tables.
//!
//! Everything here runs inside a build script, so a refusal is a panic that
//! names the file it came from.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// The value at the dotted `key`, or a panic naming `file`.
pub fn field<'a>(value: &'a Value, key: &str, file: &Path) -> &'a Value {
    key.split('.')
        .try_fold(value, |v, part| v.get(part))
        .unwrap_or_else(|| panic!("{}: missing {key}", file.display()))
}

/// Insert every leaf path of `value` below `path` ("" for the root) into
/// `paths`.
pub fn leaf_paths(value: &Value, path: &str, paths: &mut BTreeSet<String>) {
    if let Some(object) = value.as_object() {
        for (key, value) in object {
            leaf_paths(
                value,
                &if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                },
                paths,
            );
        }
    } else {
        paths.insert(path.to_owned());
    }
}

/// Refuse a key of the object `value` that `allowed` does not list.
pub fn keys(value: &Value, allowed: &[&str], path: &Path) {
    for key in value
        .as_object()
        .unwrap_or_else(|| panic!("{}: expected object", path.display()))
        .keys()
    {
        assert!(
            allowed.contains(&key.as_str()),
            "{}: unsupported key {key}",
            path.display()
        );
    }
}

/// Refuse a geometry id or kernel tag that is not `[a-z][a-z0-9_]*`.
pub fn validate_name(value: &str, kind: &str) {
    assert!(
        value.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
            && value
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_'),
        "invalid {kind} {value:?}: expected [a-z][a-z0-9_]*"
    );
}

/// The rules a declaration (a geometry directory or one geometry entry)
/// applies: `base` with its semantic rules and recipe classes narrowed,
/// never widened.
pub fn geometry_rules(base: &Value, declaration: &Value, file: &Path) -> Value {
    let mut rules = base.clone();
    if let Some(semantic) = declaration.get("semantic_rules") {
        for (key, allowed) in semantic.as_object().unwrap() {
            let supported = base["semantic_rules"]
                .get(key)
                .unwrap_or_else(|| panic!("{}: unknown semantic rule {key}", file.display()));
            assert!(
                allowed.as_array().is_some_and(|values| !values.is_empty()
                    && values
                        .iter()
                        .all(|value| supported.as_array().unwrap().contains(value))),
                "{}: unsupported semantic rule {key}",
                file.display()
            );
            rules["semantic_rules"][key] = allowed.clone();
        }
    }
    if let Some(classes) = declaration.get("quantization_recipe_classes") {
        assert!(
            classes.as_array().is_some_and(|values| !values.is_empty()
                && values
                    .iter()
                    .all(|value| base["quantization_recipe_classes"]
                        .as_array()
                        .unwrap()
                        .contains(value))),
            "{}: unsupported recipe classes",
            file.display()
        );
        rules["quantization_recipe_classes"] = classes.clone();
    }
    rules
}

/// The Rust type a field table entry declares.
pub trait RustType {
    fn rust_type(&self) -> String;
}

/// A Rust type written out in the table.
impl RustType for &str {
    fn rust_type(&self) -> String {
        (*self).to_owned()
    }
}

/// How a configuration value becomes a Rust literal.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    U32,
    Bool,
    F64,
    Str,
    /// `null` or an unsigned value.
    OptionalU32,
    U32List,
    /// A list of exactly this many unsigned values, as an array.
    U32Array(usize),
    /// A list of names, as a slice of the named enum's variants:
    /// (enum, [(configuration name, variant)]).
    Names(&'static str, &'static [(&'static str, &'static str)]),
}

impl RustType for Kind {
    fn rust_type(&self) -> String {
        match self {
            Self::U32 => "u32".into(),
            Self::Bool => "bool".into(),
            Self::F64 => "f64".into(),
            Self::Str => "&'static str".into(),
            Self::OptionalU32 => "Option<u32>".into(),
            Self::U32List => "&'static [u32]".into(),
            Self::U32Array(length) => format!("[u32; {length}]"),
            Self::Names(ty, _) => format!("&'static [{ty}]"),
        }
    }
}

impl Kind {
    /// The literal for `value`, read from `key` of `file`.
    pub fn literal(self, value: &Value, key: &str, file: &Path) -> String {
        let unsigned = |v: &Value| v.as_u64().is_some_and(|n| u32::try_from(n).is_ok());
        let unsigned_list = |v: &Value| {
            v.as_array().map(|list| {
                list.iter()
                    .map(|v| unsigned(v).then(|| v.to_string()))
                    .collect::<Option<Vec<_>>>()
            })
        };
        let literal = match self {
            Self::U32 => unsigned(value).then(|| value.to_string()),
            Self::Bool => value.is_boolean().then(|| value.to_string()),
            Self::F64 => value
                .as_f64()
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:?}")),
            Self::Str => value.is_string().then(|| value.to_string()),
            Self::OptionalU32 if value.is_null() => Some("None".into()),
            Self::OptionalU32 => unsigned(value).then(|| format!("Some({value})")),
            Self::U32List => unsigned_list(value)
                .flatten()
                .map(|list| format!("&[{}]", list.join(","))),
            Self::U32Array(length) => unsigned_list(value)
                .flatten()
                .filter(|list| list.len() == length)
                .map(|list| format!("[{}]", list.join(","))),
            Self::Names(ty, names) => value
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(|v| {
                            names
                                .iter()
                                .find(|(name, _)| v.as_str() == Some(name))
                                .map(|(_, variant)| format!("{ty}::{variant}"))
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .map(|list| format!("&[{}]", list.join(","))),
        };
        literal.unwrap_or_else(|| panic!("{}: invalid type for {key}", file.display()))
    }
}

/// A model's field and group tables.
///
/// `fields` holds (generated slot, configuration path, kind). An empty slot
/// is read and checked but not generated; an empty path is a slot the model
/// derives itself. `groups` holds (slot prefix, struct name); each group
/// becomes one struct holding its direct fields and direct subgroups.
pub struct Schema<'a, K> {
    pub fields: &'a [(&'a str, &'a str, K)],
    pub groups: &'a [(&'a str, &'a str)],
}

impl<K: RustType> Schema<'_, K> {
    /// One `#[derive(..)] pub struct` per group, fields in table order.
    pub fn declarations(&self) -> String {
        let mut output = String::new();
        for &(prefix, ty) in self.groups {
            output += &format!("#[derive(Clone, Copy, Debug, PartialEq)] pub struct {ty} {{");
            for (slot, _, kind) in self.fields {
                if let Some(name) = slot.strip_prefix(&format!("{prefix}."))
                    && !name.contains('.')
                {
                    output += &format!("pub {name}: {},", kind.rust_type());
                }
            }
            for &(group, group_ty) in self.groups {
                if let Some(name) = group.strip_prefix(&format!("{prefix}."))
                    && !name.contains('.')
                {
                    output += &format!("pub {name}: {group_ty},");
                }
            }
            output += "}\n";
        }
        output
    }

    /// The struct literal of group `prefix` from (slot, literal) pairs, in
    /// their given order, then its subgroups in table order.
    pub fn grouped_value(&self, prefix: &str, values: &[(String, String)]) -> String {
        let ty = self
            .groups
            .iter()
            .find(|(group, _)| *group == prefix)
            .unwrap()
            .1;
        let mut output = format!("{ty} {{");
        for (slot, value) in values {
            if let Some(name) = slot.strip_prefix(&format!("{prefix}."))
                && !name.contains('.')
            {
                output += &format!("{name}: {value},");
            }
        }
        for &(group, _) in self.groups {
            if let Some(name) = group.strip_prefix(&format!("{prefix}."))
                && !name.contains('.')
            {
                output += &format!("{name}: {},", self.grouped_value(group, values));
            }
        }
        output + "}"
    }

    /// The configuration paths the table reads, in table order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.fields
            .iter()
            .map(|(_, path, _)| *path)
            .filter(|path| !path.is_empty())
    }

    /// `COMPILE_FIELDS`: the table's paths and the base directory's
    /// semantic rule paths, sorted.
    pub fn compile_fields(&self, base: &Value) -> String {
        let paths = self
            .paths()
            .chain(
                base["semantic_rules"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str),
            )
            .collect::<BTreeSet<_>>();
        format!(
            "pub const COMPILE_FIELDS: &[&str] = &{:?};\n",
            paths.into_iter().collect::<Vec<_>>()
        )
    }

    /// Refuse a document whose leaf paths are not exactly the table's
    /// paths, the semantic rule paths and the runtime field paths of
    /// `rules`.
    pub fn check_leaves(&self, config: &Value, rules: &Value, path: &Path) {
        let mut leaves = BTreeSet::new();
        leaf_paths(config, "", &mut leaves);
        let expected = self
            .paths()
            .map(str::to_owned)
            .chain(rules["semantic_rules"].as_object().unwrap().keys().cloned())
            .chain(rules["runtime_fields"].as_object().unwrap().keys().cloned())
            .collect::<BTreeSet<_>>();
        for key in &expected {
            field(config, key, path);
        }
        assert_eq!(
            leaves,
            expected,
            "{}: execution field set differs",
            path.display()
        );
    }
}

impl Schema<'_, Kind> {
    /// The literal of every generated slot whose path the table reads,
    /// by slot.
    pub fn literals(&self, config: &Value, path: &Path) -> BTreeMap<String, String> {
        let mut values = BTreeMap::new();
        for &(slot, key, kind) in self.fields {
            if key.is_empty() {
                continue;
            }
            let value = kind.literal(field(config, key, path), key, path);
            if !slot.is_empty() {
                values.insert(slot.to_owned(), value);
            }
        }
        values
    }
}

/// One geometry directory: `supported.json` and the geometries it declares.
pub struct Directory {
    pub path: PathBuf,
    /// The text of `supported.json`.
    pub text: String,
    pub supported: Value,
    /// The base directory's rules narrowed by this directory's.
    pub rules: Value,
    pub geometries: Vec<Declared>,
}

/// One declared geometry with its document read.
pub struct Declared {
    /// Its `supported.json` entry.
    pub declaration: Value,
    pub id: String,
    /// The directory's rules narrowed by the entry's.
    pub rules: Value,
    pub path: PathBuf,
    pub document: String,
    pub config: Value,
}

/// The directories a contract reads its geometries from.
pub struct Catalog<'a> {
    /// The variable listing extra directories after the base
    /// `<manifest>/../geometries`.
    pub variable: &'a str,
    /// The keys `supported.json` may hold.
    pub directory_keys: &'a [&'a str],
    /// The keys a geometry entry may hold.
    pub geometry_keys: &'a [&'a str],
}

/// The text of the JSON file at `path` and its value.
pub fn read_json(path: &Path) -> (String, Value) {
    let document = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let value =
        serde_json::from_str(&document).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (document, value)
}

impl Catalog<'_> {
    /// Read every directory, its rules and its geometry documents, refusing
    /// unknown keys, rules that widen the base directory's, duplicate ids
    /// and geometry files outside their directory. Prints the build
    /// script's rerun triggers.
    pub fn read(&self) -> Vec<Directory> {
        println!("cargo:rerun-if-env-changed={}", self.variable);
        let mut paths =
            vec![PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../geometries")];
        if let Some(extra) = env::var_os(self.variable) {
            paths.extend(env::split_paths(&extra).filter(|path| !path.as_os_str().is_empty()));
        }
        let mut ids = BTreeSet::new();
        let mut directories: Vec<Directory> = Vec::new();
        for dir in paths {
            println!("cargo:rerun-if-changed={}", dir.display());
            for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
                println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
            }
            let supported_path = dir.join("supported.json");
            let (text, supported) = read_json(&supported_path);
            keys(&supported, self.directory_keys, &supported_path);
            let base = directories
                .first()
                .map_or(&supported, |base| &base.supported);
            // Every geometry admits through the base directory's default.
            let default = crate::geometry::QUANT_METHOD_DEFAULT;
            assert!(
                supported
                    .get(default)
                    .is_none_or(|value| Some(value) == base.get(default)),
                "{}: differing {default}",
                supported_path.display()
            );
            // Every geometry runs on the base directory's engines.
            let domain = crate::geometry::ENGINE_DOMAIN;
            if let Some(entries) = supported.get(domain) {
                assert!(
                    Some(entries) == base.get(domain),
                    "{}: differing {domain}",
                    supported_path.display()
                );
                engine_domain(entries, &supported_path);
            }
            let rules = geometry_rules(base, &supported, &supported_path);
            let mut geometries = Vec::new();
            for declaration in supported["geometries"]
                .as_array()
                .unwrap_or_else(|| panic!("{}: missing geometries", supported_path.display()))
            {
                keys(declaration, self.geometry_keys, &supported_path);
                let id = declaration["id"]
                    .as_str()
                    .unwrap_or_else(|| panic!("{}: missing geometry id", supported_path.display()))
                    .to_owned();
                assert!(ids.insert(id.clone()), "duplicate geometry id {id}");
                let file = declaration["file"].as_str().unwrap_or_else(|| {
                    panic!("{}: missing file for {id}", supported_path.display())
                });
                assert!(
                    Path::new(file)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_))),
                    "invalid geometry file {file}"
                );
                let path = dir.join(file);
                println!("cargo:rerun-if-changed={}", path.display());
                let (document, config) = read_json(&path);
                geometries.push(Declared {
                    rules: geometry_rules(&rules, declaration, &supported_path),
                    declaration: declaration.clone(),
                    id,
                    path,
                    document,
                    config,
                });
            }
            directories.push(Directory {
                path: dir,
                text,
                supported,
                rules,
                geometries,
            });
        }
        directories
    }
}

/// Check the form of an `engine_domain` (see
/// [`crate::geometry::ENGINE_DOMAIN`]): a path maps to one entry or a
/// non-empty list of them, each entry states exactly one well-formed limit
/// (see [`crate::geometry::engine_limit`]), a checked limit gives its
/// reason, a source cites its text, and a phase is a name.
pub fn engine_domain(domain: &Value, file: &Path) {
    let paths = domain
        .as_object()
        .unwrap_or_else(|| panic!("{}: engine_domain must be an object", file.display()));
    for (path, entries) in paths {
        let invalid =
            |what: &str| -> ! { panic!("{}: engine_domain {path}: {what}", file.display()) };
        if entries.as_array().is_some_and(Vec::is_empty) {
            invalid("a list of limits must not be empty");
        }
    }
    for (path, entry) in crate::geometry::engine_entries(domain) {
        let invalid =
            |what: &str| -> ! { panic!("{}: engine_domain {path}: {what}", file.display()) };
        if !entry.is_object() {
            invalid("each limit must be an object");
        }
        keys(
            entry,
            &[
                "values",
                "min",
                "max",
                "multiple_of",
                "max_length",
                "relation",
                "reason",
                "source",
                "source_text",
                "phase",
            ],
            file,
        );
        if crate::geometry::engine_phase(entry).is_none() {
            invalid("phase must be a non-empty string");
        }
        let limit = crate::geometry::engine_limit(entry).unwrap_or_else(|what| invalid(&what));
        let text = |key: &str| {
            entry
                .get(key)
                .map(|value| value.as_str().is_some_and(|text| !text.is_empty()))
        };
        match text("reason") {
            Some(false) => invalid("reason must be a non-empty string"),
            None if limit != crate::geometry::EngineLimit::Relation => invalid("missing reason"),
            _ => {}
        }
        if text("source") != text("source_text") || text("source") == Some(false) {
            invalid("source and source_text must be strings given together");
        }
    }
}

/// Every directory's alias lists, each list in first-seen order without
/// repeats.
pub fn merged_aliases(directories: &[Directory]) -> serde_json::Map<String, Value> {
    let mut aliases = serde_json::Map::new();
    for directory in directories {
        let Some(lists) = directory.supported["aliases"].as_object() else {
            continue;
        };
        for (name, list) in lists {
            let combined = aliases
                .entry(name.clone())
                .or_insert_with(|| Value::Array(Vec::new()))
                .as_array_mut()
                .unwrap();
            for alias in list.as_array().unwrap() {
                if !combined.contains(alias) {
                    combined.push(alias.clone());
                }
            }
        }
    }
    aliases
}

/// One `pub const <NAME>_ALIASES: &[&str]` per alias list.
pub fn render_aliases(aliases: &serde_json::Map<String, Value>) -> String {
    let mut output = String::new();
    for (name, list) in aliases {
        output += &format!(
            "pub const {}_ALIASES: &[&str] = &{:?};\n",
            name.to_uppercase(),
            list.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>()
        );
    }
    output
}

/// A supported runtime field: (configuration path, `RuntimeValues` member,
/// the configuration path of the step every value is a multiple of).
pub type RuntimeFieldSpec = (&'static str, &'static str, &'static str);

/// `RuntimeLimits { .. }` members: each supported field's compiled limit
/// from the geometry entry's `limits`.
pub fn runtime_limits(supported: &[RuntimeFieldSpec], declaration: &Value) -> String {
    supported
        .iter()
        .map(|(name, member, _)| format!("{member}: {}", declaration["limits"][name]))
        .collect::<Vec<_>>()
        .join(",")
}

/// `RuntimeField { .. }` values: each declared runtime field, with the
/// geometry's compiled limit as its maximum and the geometry's own value as
/// the qualified default.
pub fn runtime_fields(
    supported: &[RuntimeFieldSpec],
    geometry: &Declared,
    rules: &Value,
) -> String {
    let (id, path, config) = (&geometry.id, &geometry.path, &geometry.config);
    let declared = rules["runtime_fields"]
        .as_object()
        .expect("runtime field declarations");
    assert_eq!(
        declared.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        supported.iter().map(|(name, ..)| *name).collect(),
        "{id}: runtime fields differ from the supported set"
    );
    declared
        .iter()
        .map(|(name, declaration)| {
            let overridable = declaration["overridable"].as_bool().expect("overridable flag");
            let minimum = declaration["minimum"].as_u64().expect("runtime field minimum");
            let limit = geometry.declaration["limits"][name]
                .as_u64()
                .unwrap_or_else(|| panic!("{id}: missing limit for {name}"));
            let qualified = field(config, name, path).as_u64().expect("unsigned runtime field");
            let step_path = supported.iter().find(|(field, ..)| field == name).unwrap().2;
            let step = field(config, step_path, path).as_u64().expect("unsigned runtime step");
            assert!(
                minimum > 0
                    && minimum <= qualified
                    && qualified <= limit
                    && u32::try_from(limit).is_ok(),
                "{id}: {name} must satisfy 0 < minimum <= value <= compiled limit"
            );
            assert!(
                step > 0 && [minimum, qualified, limit].iter().all(|v| v.is_multiple_of(step)),
                "{id}: {name} minimum, value and compiled limit must be multiples of {step_path}"
            );
            format!(
                "RuntimeField {{ name: {name:?}, overridable: {overridable}, minimum: {minimum}, maximum: {limit}, step: {step}, qualified: {qualified} }}"
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Each geometry's tuning table: `<tuning>/<id>/schedules.json` where the
/// geometry has one, as (geometry id, text).
#[derive(Default)]
pub struct TuningTables(Vec<(String, String)>);

impl TuningTables {
    /// Read `<tuning>/<id>/schedules.json` if the geometry has one.
    pub fn add(&mut self, id: &str, tuning: &Path) {
        if tuning.exists() {
            println!("cargo:rerun-if-changed={}", tuning.display());
        }
        let dir = tuning.join(id);
        if !dir.exists() {
            return;
        }
        println!("cargo:rerun-if-changed={}", dir.display());
        let path = dir.join("schedules.json");
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        self.0.push((id.to_owned(), text));
    }

    /// The generated `TABLES` item.
    pub fn render(&self) -> String {
        format!(
            "pub(crate) static TABLES: &[(&str, &str)] = &{:?};\n",
            self.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(result: std::thread::Result<()>) -> String {
        let error = result.unwrap_err();
        error
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| error.downcast_ref::<&str>().unwrap().to_string())
    }

    #[test]
    fn missing_paths_and_unknown_keys_name_the_source() {
        let path = Path::new("shape.json");
        assert_eq!(
            message(std::panic::catch_unwind(|| {
                field(&json!({"a": {}}), "a.b", path);
            })),
            "shape.json: missing a.b"
        );
        assert_eq!(
            message(std::panic::catch_unwind(|| keys(
                &json!({"bogus": 1}),
                &["id"],
                path
            ))),
            "shape.json: unsupported key bogus"
        );
        let mut leaves = BTreeSet::new();
        leaf_paths(
            &json!({"a": {"b": 1, "c": [1]}, "d": null}),
            "",
            &mut leaves,
        );
        assert_eq!(
            leaves,
            BTreeSet::from(["a.b".into(), "a.c".into(), "d".into()])
        );
    }

    #[test]
    fn rules_narrow_never_widen() {
        let base = json!({"semantic_rules": {"dtype": ["bfloat16", "float16"]},
            "quantization_recipe_classes": ["fp8", "modelopt"]});
        let path = Path::new("supported.json");
        let narrowed = geometry_rules(
            &base,
            &json!({"semantic_rules": {"dtype": ["bfloat16"]},
                "quantization_recipe_classes": ["fp8"]}),
            path,
        );
        assert_eq!(narrowed["semantic_rules"]["dtype"], json!(["bfloat16"]));
        assert_eq!(narrowed["quantization_recipe_classes"], json!(["fp8"]));
        for declaration in [
            json!({"semantic_rules": {"dtype": ["float32"]}}),
            json!({"semantic_rules": {"dtype": []}}),
            json!({"semantic_rules": {"unknown": [true]}}),
            json!({"quantization_recipe_classes": ["awq"]}),
        ] {
            assert!(
                std::panic::catch_unwind(|| geometry_rules(&base, &declaration, path)).is_err()
            );
        }
    }

    #[test]
    fn engine_domain_entries_state_one_limit() {
        let file = Path::new("supported.json");
        let cited = |entry: Value| {
            let mut entry = entry;
            entry["source"] = "a.rs".into();
            entry["source_text"] = "n <= 8".into();
            entry
        };
        engine_domain(
            &json!({
                "a": {"values": [1, 2], "reason": "r"},
                "b": cited(json!({"max": 8, "reason": "r"})),
                "c": {"min": 2, "max": 8, "multiple_of": 2, "reason": "r"},
                "d": {"multiple_of": 64, "reason": "r"},
                "e": {"max_length": 1, "reason": "r"},
                "f": cited(json!({"relation": "x < y"})),
                "g": [
                    {"min": 2, "max": 8, "reason": "r"},
                    {"values": [4], "phase": "late", "reason": "r"},
                ],
                "h": [{"multiple_of": 4, "phase": "late", "reason": "r"}],
            }),
            file,
        );
        for (entry, expected) in [
            (
                json!({"values": [1], "max": 2, "reason": "r"}),
                "expected one of",
            ),
            (json!({"reason": "r"}), "expected one of"),
            (json!({"values": [], "reason": "r"}), "non-empty"),
            (json!({"max": -1, "reason": "r"}), "unsigned"),
            (json!({"min": 9, "max": 8, "reason": "r"}), "empty range"),
            (json!({"multiple_of": 0, "reason": "r"}), "empty range"),
            (json!({"values": [1]}), "missing reason"),
            (
                json!({"values": [1], "reason": "r", "source": "a.rs"}),
                "together",
            ),
            (json!({"values": [1], "reason": "r", "limit": 1}), "limit"),
            (json!([]), "must not be empty"),
            (json!([{"values": [1], "reason": "r"}, 8]), "an object"),
            (
                json!([{"values": [1], "reason": "r", "phase": ""}]),
                "phase must be",
            ),
            (json!([{"values": [1]}]), "missing reason"),
        ] {
            let error = message(std::panic::catch_unwind(|| {
                engine_domain(&json!({ "k": entry }), file)
            }));
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn engine_domain_refuses_wrong_types_in_every_entry_kind() {
        let file = Path::new("supported.json");
        let mut accepted = Vec::new();
        for entry in [
            json!(8),
            json!({"values": 128, "reason": "r"}),
            json!({"values": [128, null], "reason": "r"}),
            json!({"values": [128, "128"], "reason": "r"}),
            json!({"values": [128, false], "reason": "r"}),
            json!({"values": [128, []], "reason": "r"}),
            json!({"values": [128, {}], "reason": "r"}),
            json!({"max": 8.0, "reason": "r"}),
            json!({"min": "2", "reason": "r"}),
            json!({"multiple_of": null, "reason": "r"}),
            json!({"min": 2, "max": 3, "multiple_of": 4, "reason": "r"}),
            json!({"max_length": null, "reason": "r"}),
            json!({"max_length": 1.5, "reason": "r"}),
            json!({"relation": 5}),
            json!({"relation": ""}),
            json!({"relation": "x < y", "reason": 5}),
            json!({"values": [1], "reason": ""}),
            json!({"values": [1], "reason": "r", "source": 1, "source_text": 1}),
            json!({"values": [1], "reason": "r", "phase": 5}),
            json!({"values": [1], "reason": "r", "phase": null}),
            json!([]),
            json!([8]),
            json!([[{"values": [1], "reason": "r"}]]),
            json!([{"values": [1], "reason": "r"}, {"values": [128, null], "reason": "r"}]),
        ] {
            let domain = json!({ "k": entry });
            if std::panic::catch_unwind(|| engine_domain(&domain, file)).is_ok() {
                accepted.push(domain);
            }
        }
        assert!(accepted.is_empty(), "accepted {accepted:?}");
    }

    #[test]
    fn literals_check_their_kind() {
        let file = Path::new("geometry.json");
        let layers = Kind::Names("LayerKind", &[("a", "A"), ("b", "B")]);
        assert_eq!(Kind::U32.literal(&json!(7), "k", file), "7");
        assert_eq!(Kind::F64.literal(&json!(1), "k", file), "1.0");
        assert_eq!(
            Kind::F64.literal(&serde_json::from_str("1e7").unwrap(), "k", file),
            "10000000.0"
        );
        assert_eq!(Kind::Str.literal(&json!("x"), "k", file), "\"x\"");
        assert_eq!(Kind::OptionalU32.literal(&Value::Null, "k", file), "None");
        assert_eq!(Kind::OptionalU32.literal(&json!(3), "k", file), "Some(3)");
        assert_eq!(Kind::U32List.literal(&json!([1, 2]), "k", file), "&[1,2]");
        assert_eq!(
            Kind::U32Array(2).literal(&json!([1, 2]), "k", file),
            "[1,2]"
        );
        assert_eq!(
            layers.literal(&json!(["b", "a"]), "k", file),
            "&[LayerKind::B,LayerKind::A]"
        );
        assert_eq!(layers.rust_type(), "&'static [LayerKind]");
        assert_eq!(
            message(std::panic::catch_unwind(|| {
                Kind::U32.literal(&json!(4294967296_u64), "hidden", file);
            })),
            "geometry.json: invalid type for hidden"
        );
        for (kind, value) in [
            (Kind::U32, json!(-1)),
            (Kind::Bool, json!(1)),
            (Kind::F64, json!("1")),
            (Kind::OptionalU32, json!(-1)),
            (Kind::U32List, json!([1, -1])),
            (Kind::U32Array(3), json!([1, 2])),
            (layers, json!(["c"])),
        ] {
            assert!(std::panic::catch_unwind(|| kind.literal(&value, "k", file)).is_err());
        }
    }

    #[test]
    fn schema_renders_nested_groups() {
        let schema = Schema {
            fields: &[
                ("shape.hidden", "hidden_size", Kind::U32),
                ("shape.moe.experts", "moe.experts", Kind::U32),
                ("", "checked_only", Kind::Bool),
                ("semantics.rotary", "", Kind::U32),
            ],
            groups: &[
                ("shape", "Shape"),
                ("shape.moe", "Moe"),
                ("semantics", "Semantics"),
            ],
        };
        assert_eq!(
            schema.declarations(),
            "#[derive(Clone, Copy, Debug, PartialEq)] pub struct Shape {pub hidden: u32,pub moe: Moe,}\n\
             #[derive(Clone, Copy, Debug, PartialEq)] pub struct Moe {pub experts: u32,}\n\
             #[derive(Clone, Copy, Debug, PartialEq)] pub struct Semantics {pub rotary: u32,}\n"
        );
        let config =
            json!({"hidden_size": 8, "moe": {"experts": 4}, "checked_only": true, "rule": "x"});
        let path = Path::new("geometry.json");
        let values = schema
            .literals(&config, path)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            schema.grouped_value("shape", &values),
            "Shape {hidden: 8,moe: Moe {experts: 4,},}"
        );
        let base = json!({"semantic_rules": {"rule": ["x"]}, "runtime_fields": {}});
        assert_eq!(
            schema.compile_fields(&base),
            "pub const COMPILE_FIELDS: &[&str] = &[\"checked_only\", \"hidden_size\", \"moe.experts\", \"rule\"];\n"
        );
        schema.check_leaves(&config, &base, path);
        let mut extra = config.clone();
        extra["unexpected"] = json!(1);
        assert!(std::panic::catch_unwind(|| schema.check_leaves(&extra, &base, path)).is_err());
    }

    #[test]
    fn runtime_fields_stay_within_compiled_limits() {
        const SUPPORTED: &[RuntimeFieldSpec] = &[("topk", "topk", "pool")];
        let geometry = |value: u64, limit: u64| Declared {
            declaration: json!({"limits": {"topk": limit}}),
            id: "g".into(),
            rules: Value::Null,
            path: "g.json".into(),
            document: String::new(),
            config: json!({"topk": value, "pool": 4}),
        };
        let rules = json!({"runtime_fields": {"topk": {"overridable": true, "minimum": 4}}});
        assert_eq!(
            runtime_fields(SUPPORTED, &geometry(8, 16), &rules),
            "RuntimeField { name: \"topk\", overridable: true, minimum: 4, maximum: 16, step: 4, qualified: 8 }"
        );
        assert_eq!(
            runtime_limits(SUPPORTED, &geometry(8, 16).declaration),
            "topk: 16"
        );
        for (value, limit) in [(32, 16), (6, 16), (8, 18)] {
            assert!(
                std::panic::catch_unwind(|| runtime_fields(
                    SUPPORTED,
                    &geometry(value, limit),
                    &rules
                ))
                .is_err()
            );
        }
        assert!(
            std::panic::catch_unwind(|| runtime_fields(
                SUPPORTED,
                &geometry(8, 16),
                &json!({"runtime_fields": {}})
            ))
            .is_err()
        );
    }

    #[test]
    fn aliases_merge_in_first_seen_order() {
        let directory = |aliases: Value| Directory {
            path: PathBuf::new(),
            text: String::new(),
            supported: json!({"aliases": aliases}),
            rules: Value::Null,
            geometries: Vec::new(),
        };
        let merged = merged_aliases(&[
            directory(json!({"model_type": ["a", "b"]})),
            directory(json!({"model_type": ["b", "c"], "architecture": ["X"]})),
        ]);
        assert_eq!(
            render_aliases(&merged),
            "pub const MODEL_TYPE_ALIASES: &[&str] = &[\"a\", \"b\", \"c\"];\n\
             pub const ARCHITECTURE_ALIASES: &[&str] = &[\"X\"];\n"
        );
    }
}
