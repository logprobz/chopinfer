//! Match closed execution shapes while allowing bounded runtime settings.
use serde_json::Value;

fn equal(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Number(_), Value::Number(_)) => same_number(actual, expected),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| equal(value, other)))
        }
        _ => actual == expected,
    }
}

pub fn field<'a>(document: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(document, |value, key| value.get(key))
}

/// Field order defines the first difference reported for each geometry.
pub fn match_geometry(
    config: &Value,
    geometries: &[(&str, Value)],
    compile_fields: &[&str],
) -> Result<usize, String> {
    let mut differences = Vec::new();
    for (index, (id, document)) in geometries.iter().enumerate() {
        if let Some(path) = compile_fields.iter().find(|path| {
            field(config, path)
                .zip(field(document, path))
                .is_none_or(|(a, b)| !equal(a, b))
        }) {
            let display =
                |value: Option<&Value>| value.map_or_else(|| "absent".into(), Value::to_string);
            differences.push(format!(
                "{id}: {path} expected {}, received {}",
                display(field(document, path)),
                display(field(config, path))
            ));
        } else {
            return Ok(index);
        }
    }
    Err(format!(
        "unsupported geometry: {}; supported ids: {}",
        differences.join("; "),
        geometries
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The `supported.json` key naming the recipe class admitted for a
/// quantization recipe without a string `quant_method`. Without it such a
/// recipe is refused.
pub const QUANT_METHOD_DEFAULT: &str = "quant_method_default";

/// Why a config fails one geometry's declared rules.
#[derive(Debug, PartialEq)]
pub enum Refusal<'a> {
    /// The value at `path` (absent when `None`) is not in `allowed`; a list
    /// must hold allowed items only.
    Semantic {
        path: &'a str,
        allowed: &'a [Value],
        received: Option<&'a Value>,
    },
    /// The recipe's class is not among `classes`. `method` is the recipe's
    /// raw `quant_method`; `class` is the class checked, which is the
    /// declared default when `method` is absent or not a string.
    RecipeClass {
        classes: &'a [Value],
        method: Option<&'a Value>,
        class: Option<&'a str>,
    },
}

impl std::fmt::Display for Refusal<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display =
            |value: Option<&Value>| value.map_or_else(|| "absent".into(), Value::to_string);
        match self {
            Self::Semantic {
                path,
                allowed,
                received,
            } => write!(
                f,
                "unsupported {path}: expected one of {}, received {}",
                Value::Array(allowed.to_vec()),
                display(*received)
            ),
            Self::RecipeClass {
                classes, method, ..
            } => write!(
                f,
                "unsupported quantization_config.quant_method: expected one of {}, received {}",
                Value::Array(classes.to_vec()),
                display(*method)
            ),
        }
    }
}

/// Check `config` against one geometry's declared rules: each
/// `semantic_rules` path must hold an allowed value (a list, allowed items
/// only), and the quantization `recipe` (the object holding
/// `quant_method`) must name one of `quantization_recipe_classes`, or fall
/// back to the declared `quant_method_default`. Returns the recipe class.
pub fn admit<'a>(
    config: &'a Value,
    recipe: Option<&'a Value>,
    rules: &'a Value,
) -> Result<&'a str, Refusal<'a>> {
    for (path, allowed) in rules["semantic_rules"].as_object().unwrap() {
        let allowed = allowed.as_array().unwrap();
        let received = field(config, path);
        let accepted = received.is_some_and(|actual| match actual.as_array() {
            Some(list) => list.iter().all(|item| allowed.contains(item)),
            None => allowed.contains(actual),
        });
        if !accepted {
            return Err(Refusal::Semantic {
                path,
                allowed,
                received,
            });
        }
    }
    let method = recipe.and_then(|recipe| recipe.get("quant_method"));
    let class = method
        .and_then(Value::as_str)
        .or_else(|| rules[QUANT_METHOD_DEFAULT].as_str());
    let classes = rules["quantization_recipe_classes"].as_array().unwrap();
    match class {
        Some(class) if classes.iter().any(|listed| listed == class) => Ok(class),
        _ => Err(Refusal::RecipeClass {
            classes,
            method,
            class,
        }),
    }
}

/// The `supported.json` key declaring the shared engines' limits on config
/// values. Each entry maps a config path to one limit, or to a list of
/// limits: `values` (the allowed numbers), a range (`max`, `min` defaulting
/// to 1, `multiple_of`), `max_length` (a list's length), or a `relation`
/// its model checks in code. `reason` explains a checked limit; `source`
/// and `source_text` cite the engine code stating it. `phase` names the
/// point in its model's checks where a limit runs ([`FIRST_ENGINE_PHASE`]
/// when absent); see [`check_engine_phase`]. Models check each phase's
/// entries in declaration order, so the first refusal does not depend on
/// code order.
pub const ENGINE_DOMAIN: &str = "engine_domain";

/// The phase of an `engine_domain` entry without `phase`: checked before
/// its model's code relations.
pub const FIRST_ENGINE_PHASE: &str = "before_relations";

/// A config value outside one `engine_domain` entry.
#[derive(Debug, PartialEq)]
pub struct DomainRefusal<'a> {
    pub path: &'a str,
    pub entry: &'a Value,
    /// The value checked: `actual`, or a list's length for `max_length`.
    pub received: Value,
    pub cause: DomainCause,
}

/// Why an `engine_domain` entry refused a value.
#[derive(Debug, PartialEq)]
pub enum DomainCause {
    /// The value is outside the entry's limit.
    Outside,
    /// The entry states no well-formed limit (see [`engine_limit`]).
    Malformed(String),
    /// The entry's phase is not one its model checks, or not a name.
    Phase,
}

impl DomainRefusal<'_> {
    /// The restriction to report: the entry's `reason`, or why the entry
    /// refused when it has none (a relation needs no reason). Never empty.
    pub fn restriction(&self) -> String {
        let reason = self.entry.get("reason").and_then(Value::as_str);
        if let Some(reason) = reason.filter(|reason| !reason.is_empty()) {
            return reason.to_owned();
        }
        match &self.cause {
            DomainCause::Outside => format!("engine_domain limit {}", self.entry),
            DomainCause::Malformed(why) => format!("malformed engine_domain entry: {why}"),
            DomainCause::Phase => format!(
                "engine_domain phase {} is not checked",
                self.entry
                    .get("phase")
                    .map_or_else(|| format!("{FIRST_ENGINE_PHASE:?}"), Value::to_string)
            ),
        }
    }
}

/// The limit one `engine_domain` entry states.
#[derive(Debug, PartialEq)]
pub enum EngineLimit<'a> {
    /// One of these numbers.
    Values(&'a [Value]),
    /// An unsigned integer in `min..=max` that `multiple_of` divides.
    Range {
        min: u64,
        max: u64,
        multiple_of: u64,
    },
    /// A list of at most this many elements.
    MaxLength(u64),
    /// A relation its model checks in code.
    Relation,
}

/// Read the limit `entry` states, or say why it states none: an entry
/// gives exactly one kind of limit, `values` lists numbers, bounds are
/// unsigned integers admitting at least one value, and a relation is text.
pub fn engine_limit(entry: &Value) -> Result<EngineLimit<'_>, String> {
    const RANGE: [&str; 3] = ["min", "max", "multiple_of"];
    let given = |key: &str| entry.get(key).is_some();
    let kinds = [
        given("values"),
        RANGE.iter().any(|key| given(key)),
        given("max_length"),
        given("relation"),
    ];
    if kinds.iter().filter(|&&kind| kind).count() != 1 {
        return Err("expected one of values, a range, max_length or relation".into());
    }
    let unsigned = |key: &str, default| match entry.get(key) {
        None => Ok(default),
        Some(n) => n
            .as_u64()
            .ok_or_else(|| format!("{key} must be an unsigned integer")),
    };
    if let Some(values) = entry.get("values") {
        let values = values
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or("values must be a non-empty list")?;
        if !values.iter().all(Value::is_number) {
            return Err("values must be numbers".into());
        }
        return Ok(EngineLimit::Values(values));
    }
    if given("max_length") {
        return unsigned("max_length", 0).map(EngineLimit::MaxLength);
    }
    if let Some(relation) = entry.get("relation") {
        return match relation.as_str() {
            Some(text) if !text.is_empty() => Ok(EngineLimit::Relation),
            _ => Err("relation must be a non-empty string".into()),
        };
    }
    let (min, max, multiple_of) = (
        unsigned("min", 1)?,
        unsigned("max", u64::MAX)?,
        unsigned("multiple_of", 1)?,
    );
    // The first multiple at or above `min` must not exceed `max`.
    if multiple_of == 0
        || min
            .div_ceil(multiple_of)
            .checked_mul(multiple_of)
            .is_none_or(|first| first > max)
    {
        return Err("empty range".into());
    }
    Ok(EngineLimit::Range {
        min,
        max,
        multiple_of,
    })
}

/// Whether `a` and `b` are the same number, compared exactly: integers
/// as integers, and a float equals an integer only when it is exactly that
/// integer. 128.0 matches 128, but 2^53 + 1 matches neither 2^53 nor the
/// float 2^53 it rounds to.
fn same_number(a: &Value, b: &Value) -> bool {
    let (Value::Number(a), Value::Number(b)) = (a, b) else {
        return false;
    };
    let integer = |n: &serde_json::Number| {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    let integral = |float: &serde_json::Number, n: i128| {
        float
            .as_f64()
            .is_some_and(|f| f.fract() == 0.0 && f as i128 == n)
    };
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => a == b,
        (Some(n), None) => integral(b, n),
        (None, Some(n)) => integral(a, n),
        (None, None) => a.as_f64() == b.as_f64(),
    }
}

/// Every `engine_domain` entry with its config path, in declaration order;
/// a path declaring a list of limits yields each in list order.
pub fn engine_entries(domain: &Value) -> impl Iterator<Item = (&str, &Value)> {
    domain
        .as_object()
        .into_iter()
        .flatten()
        .flat_map(|(path, entries)| {
            let entries = match entries {
                Value::Array(list) => list.as_slice(),
                entry => std::slice::from_ref(entry),
            };
            entries.iter().map(move |entry| (path.as_str(), entry))
        })
}

/// The phase `entry` runs in, or `None` when its `phase` is not a
/// non-empty string.
pub fn engine_phase(entry: &Value) -> Option<&str> {
    match entry.get("phase") {
        None => Some(FIRST_ENGINE_PHASE),
        Some(phase) => phase.as_str().filter(|phase| !phase.is_empty()),
    }
}

/// Check the values `config` holds against the `engine_domain` entries of
/// `phase`, in declaration order. `phases` lists every phase its model
/// checks, in the order it checks them. An entry whose phase is missing
/// from `phases` or malformed refuses every config in the first phase, so
/// a misnamed phase cannot skip its limit.
pub fn check_engine_phase<'a>(
    domain: &'a Value,
    phases: &[&str],
    phase: &str,
    config: &Value,
) -> Result<(), DomainRefusal<'a>> {
    for (path, entry) in engine_entries(domain) {
        let actual = field(config, path).unwrap_or(&Value::Null);
        match engine_phase(entry).filter(|declared| phases.contains(declared)) {
            Some(declared) if declared == phase => check_engine_limit(path, entry, actual)?,
            Some(_) => {}
            None if phases.first() == Some(&phase) => {
                return Err(DomainRefusal {
                    path,
                    entry,
                    received: actual.clone(),
                    cause: DomainCause::Phase,
                });
            }
            None => {}
        }
    }
    Ok(())
}

/// Check `actual`, the value at `path` (or one its model derives for
/// `path`), against its `engine_domain` entry. A `relation` is the
/// model's to check and always passes here; a malformed entry (see
/// [`engine_limit`]) refuses every value. The entry's `phase` is the
/// caller's to honor.
pub fn check_engine_limit<'a>(
    path: &'a str,
    entry: &'a Value,
    actual: &Value,
) -> Result<(), DomainRefusal<'a>> {
    let refusal = |received: Value, cause| DomainRefusal {
        path,
        entry,
        received,
        cause,
    };
    let accepted = match engine_limit(entry) {
        Err(why) => return Err(refusal(actual.clone(), DomainCause::Malformed(why))),
        Ok(EngineLimit::Relation) => true,
        Ok(EngineLimit::Values(values)) => values.iter().any(|value| same_number(value, actual)),
        Ok(EngineLimit::MaxLength(limit)) => {
            let length = actual.as_array().map(Vec::len);
            if length.is_some_and(|n| n as u64 <= limit) {
                return Ok(());
            }
            return Err(refusal(
                length.map_or(Value::Null, Value::from),
                DomainCause::Outside,
            ));
        }
        Ok(EngineLimit::Range {
            min,
            max,
            multiple_of,
        }) => actual
            .as_u64()
            .is_some_and(|n| (min..=max).contains(&n) && n.is_multiple_of(multiple_of)),
    };
    if accepted {
        Ok(())
    } else {
        Err(refusal(actual.clone(), DomainCause::Outside))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn match_closed_fields() {
        let shapes = [
            ("small", json!({"width":8})),
            ("large", json!({"width":16})),
        ];
        assert_eq!(
            match_geometry(&json!({"width":16,"label":"new"}), &shapes, &["width"]),
            Ok(1)
        );
        assert_eq!(match_geometry(&json!({}), &shapes, &["width"]), Err("unsupported geometry: small: width expected 8, received absent; large: width expected 16, received absent; supported ids: small, large".into()));
        assert_eq!(match_geometry(&json!({"extra":1}), &[("empty", json!({}))], &["extra"]), Err("unsupported geometry: empty: extra expected absent, received 1; supported ids: empty".into()));
    }

    #[test]
    fn compare_nested_numbers_and_float_arrays() {
        assert!(equal(
            &json!([{"x":[1, 0.25]}]),
            &json!([{"x":[1.0, 0.25]}])
        ));
        assert!(!equal(
            &json!(9007199254740993_u64),
            &json!(9007199254740992_u64)
        ));
        assert!(!equal(&json!([0.25]), &json!([0.5])));
    }

    #[test]
    fn closed_fields_compare_integers_exactly() {
        let (int, float) = (json!(9007199254740993_u64), json!(9007199254740992.0));
        let matched = [
            ("scalar", equal(&int, &float)),
            ("reversed", equal(&float, &int)),
            (
                "array",
                equal(&json!([{"x": [int]}]), &json!([{"x": [float]}])),
            ),
            (
                "object",
                equal(&json!({"x": {"y": float}}), &json!({"x": {"y": int}})),
            ),
            (
                "match_geometry",
                match_geometry(&json!({"w": int}), &[("g", json!({"w": float}))], &["w"]).is_ok(),
            ),
        ];
        let wrong = matched
            .iter()
            .filter(|(_, matched)| *matched)
            .map(|(case, _)| *case)
            .collect::<Vec<_>>();
        assert!(
            wrong.is_empty(),
            "2^53 + 1 matched 2^53 as a float: {wrong:?}"
        );
        // An integral float still equals its integer.
        assert!(equal(&json!(128), &json!(128.0)));
        assert!(equal(&json!([{"x": 128.0}]), &json!([{"x": 128}])));
        assert!(equal(
            &json!(9007199254740992_u64),
            &json!(9007199254740992.0)
        ));
        assert_eq!(
            match_geometry(&json!({"w": 128.0}), &[("g", json!({"w": 128}))], &["w"]),
            Ok(0)
        );
    }

    #[test]
    fn admit_checks_semantics_then_the_recipe_class() {
        let rules = json!({"semantic_rules": {"act": ["silu"], "layers": ["a", "b"]},
            "quantization_recipe_classes": ["fp8", "modelopt"]});
        let config = json!({"act": "silu", "layers": ["a", "b", "a"]});
        let fp8 = json!({"quant_method": "fp8"});
        assert_eq!(admit(&config, Some(&fp8), &rules), Ok("fp8"));
        let refused = json!({"act": "relu", "layers": ["a"]});
        assert_eq!(
            admit(&refused, Some(&fp8), &rules).unwrap_err().to_string(),
            "unsupported act: expected one of [\"silu\"], received \"relu\""
        );
        let refused = json!({"act": "silu", "layers": ["c"]});
        assert!(matches!(
            admit(&refused, Some(&fp8), &rules),
            Err(Refusal::Semantic { path: "layers", .. })
        ));
        assert_eq!(
            admit(&json!({"layers": []}), None, &rules)
                .unwrap_err()
                .to_string(),
            "unsupported act: expected one of [\"silu\"], received absent"
        );
        let awq = json!({"quant_method": "awq"});
        assert_eq!(
            admit(&config, Some(&awq), &rules).unwrap_err(),
            Refusal::RecipeClass {
                classes: rules["quantization_recipe_classes"].as_array().unwrap(),
                method: Some(&awq["quant_method"]),
                class: Some("awq"),
            }
        );
    }

    #[test]
    fn engine_limits_check_values_ranges_and_lengths() {
        let check = |entry: Value, actual: Value| {
            check_engine_limit("x", &entry, &actual)
                .map_err(|refusal| refusal.received)
                .err()
        };
        let values = json!({"values": [128, 256]});
        assert_eq!(check(values.clone(), json!(256)), None);
        assert_eq!(check(values.clone(), json!(256.0)), None);
        assert_eq!(check(values.clone(), json!(64)), Some(json!(64)));
        assert_eq!(check(values, Value::Null), Some(Value::Null));
        let max = json!({"max": 8});
        assert_eq!(check(max.clone(), json!(8)), None);
        for refused in [json!(0), json!(9), json!(2.5), json!("4")] {
            assert_eq!(check(max.clone(), refused.clone()), Some(refused));
        }
        let range = json!({"min": 16, "max": 512, "multiple_of": 16});
        for (actual, accepted) in [
            (16, true),
            (320, true),
            (512, true),
            (8, false),
            (24, false),
        ] {
            assert_eq!(check(range.clone(), json!(actual)).is_none(), accepted);
        }
        let multiple = json!({"multiple_of": 64});
        assert_eq!(check(multiple.clone(), json!(1_u64 << 40)), None);
        assert_eq!(check(multiple, json!(0)), Some(json!(0)));
        let length = json!({"max_length": 1});
        assert_eq!(check(length.clone(), json!([3])), None);
        assert_eq!(check(length.clone(), json!([1, 2])), Some(json!(2)));
        assert_eq!(check(length, json!(1)), Some(Value::Null));
        assert_eq!(check(json!({"relation": "x < y"}), json!("any")), None);
    }

    #[test]
    fn engine_limits_refuse_values_of_the_wrong_type() {
        let wrong = [
            Value::Null,
            json!("8"),
            json!(true),
            json!([]),
            json!({}),
            json!(8.5),
        ];
        for entry in [
            json!({"values": [8]}),
            json!({"max": 8}),
            json!({"min": 8, "max": 16, "multiple_of": 8}),
        ] {
            for actual in wrong.iter().chain(&[json!(-8)]) {
                assert!(
                    check_engine_limit("x", &entry, actual).is_err(),
                    "{entry} {actual}"
                );
            }
        }
        // A range needs an integer, not an integral float.
        assert!(check_engine_limit("x", &json!({"max": 8}), &json!(8.0)).is_err());
        // A malformed entry refuses every value rather than matching it.
        for entry in [
            json!({"values": [8, null]}),
            json!({"values": [8, "8"]}),
            json!({"values": [8, true, [], {}]}),
            json!({"max": "8"}),
            json!({"max_length": "1"}),
            json!({"relation": 5}),
            json!({"relation": "x < y", "max": 8}),
        ] {
            for actual in wrong
                .iter()
                .chain(&[json!(9), json!([1, 2]), json!("wrong")])
            {
                assert!(
                    check_engine_limit("x", &entry, actual).is_err(),
                    "{entry} {actual}"
                );
            }
        }
    }

    #[test]
    fn engine_phases_check_their_entries_in_declaration_order() {
        let domain = json!({
            "a": {"max": 8, "reason": "r"},
            "b": [
                {"max": 8, "reason": "r"},
                {"values": [4], "phase": "late", "reason": "r"},
            ],
            "c": {"values": [1], "phase": "late", "reason": "r"},
        });
        let phases = [FIRST_ENGINE_PHASE, "late"];
        let refused = |phase: &str, config: Value| {
            check_engine_phase(&domain, &phases, phase, &config)
                .err()
                .map(|refusal| (refusal.path, refusal.received))
        };
        let config = json!({"a": 1, "b": 4, "c": 1});
        assert_eq!(refused(FIRST_ENGINE_PHASE, config.clone()), None);
        assert_eq!(refused("late", config), None);
        // A path's limits run in their own phases.
        let config = json!({"a": 1, "b": 5, "c": 2});
        assert_eq!(refused(FIRST_ENGINE_PHASE, config.clone()), None);
        assert_eq!(refused("late", config), Some(("b", json!(5))));
        let config = json!({"a": 9, "b": 9, "c": 2});
        assert_eq!(
            refused(FIRST_ENGINE_PHASE, config.clone()),
            Some(("a", json!(9)))
        );
        assert_eq!(refused("late", config), Some(("b", json!(9))));
        assert_eq!(
            refused(FIRST_ENGINE_PHASE, json!({"a": 1, "b": 9})),
            Some(("b", json!(9)))
        );
        assert_eq!(refused("late", json!({"b": 4})), Some(("c", Value::Null)));
        let entries = engine_entries(&domain)
            .map(|(path, entry)| (path, engine_phase(entry).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            entries,
            [
                ("a", FIRST_ENGINE_PHASE),
                ("b", FIRST_ENGINE_PHASE),
                ("b", "late"),
                ("c", "late")
            ]
        );
    }

    #[test]
    fn unknown_engine_phases_refuse_in_the_first_phase() {
        let config = json!({"a": 1, "b": 1});
        for entry in [
            json!({"values": [1], "phase": "later", "reason": "r"}),
            json!({"values": [1], "phase": "", "reason": "r"}),
            json!({"values": [1], "phase": 2, "reason": "r"}),
        ] {
            let domain = json!({"a": {"values": [1], "reason": "r"}, "b": [entry]});
            for phase in ["late", "x"] {
                assert_eq!(
                    check_engine_phase(&domain, &[FIRST_ENGINE_PHASE, "late"], phase, &config),
                    Ok(())
                );
            }
            let refusal = check_engine_phase(
                &domain,
                &[FIRST_ENGINE_PHASE, "late"],
                FIRST_ENGINE_PHASE,
                &config,
            )
            .unwrap_err();
            assert_eq!((refusal.path, refusal.received), ("b", json!(1)));
        }
        // A malformed list member refuses like any malformed entry.
        let domain = json!({"a": [{"values": [1], "reason": "r"}, 5]});
        assert!(
            check_engine_phase(&domain, &[FIRST_ENGINE_PHASE], FIRST_ENGINE_PHASE, &config)
                .is_err()
        );
    }

    #[test]
    fn every_entry_form_refuses_with_a_restriction_and_never_panics() {
        let kinds = [
            json!({}),
            json!({"values": [4]}),
            json!({"max": 2}),
            json!({"max_length": 1}),
            json!({"relation": "x > y"}),
            json!({"relation": 5}),
            json!({"relation": "x > y", "values": [4]}),
            json!({"values": [4, null]}),
        ];
        let optional = |values: &[Value]| {
            std::iter::once(None)
                .chain(values.iter().cloned().map(Some))
                .collect::<Vec<_>>()
        };
        let reasons = optional(&[json!("r"), json!(""), json!(5)]);
        let sources = optional(&[json!("a.rs"), json!(1)]);
        let texts = optional(&[json!("n"), json!(1)]);
        let phase_values = optional(&[
            json!(FIRST_ENGINE_PHASE),
            json!("late"),
            json!("later"),
            json!(""),
            json!(5),
            Value::Null,
        ]);
        let mut entries = vec![json!(5), Value::Null, json!("r"), json!([])];
        for kind in &kinds {
            for reason in &reasons {
                for source in &sources {
                    for text in &texts {
                        for phase in &phase_values {
                            let mut entry = kind.clone();
                            for (key, field) in [
                                ("reason", reason),
                                ("source", source),
                                ("source_text", text),
                                ("phase", phase),
                            ] {
                                if let Some(field) = field {
                                    entry[key] = field.clone();
                                }
                            }
                            entries.push(entry);
                        }
                    }
                }
            }
        }
        let phases = [FIRST_ENGINE_PHASE, "late"];
        let configs = [
            json!(4),
            json!(9),
            Value::Null,
            json!("s"),
            json!([1]),
            json!(4.0),
        ];
        let mut checked = 0;
        for entry in &entries {
            for domain in [json!({ "k": entry }), json!({ "k": [entry] })] {
                let member = match &domain["k"] {
                    Value::Array(list) => list.first(),
                    entry => Some(entry),
                };
                for actual in &configs {
                    let config = json!({ "k": actual });
                    for phase in phases {
                        let result = std::panic::catch_unwind(|| {
                            check_engine_phase(&domain, &phases, phase, &config)
                                .map_err(|refusal| (refusal.restriction(), refusal.cause))
                        });
                        let result = result.unwrap_or_else(|_| panic!("{domain} {config} {phase}"));
                        checked += 1;
                        let Some(member) = member else {
                            assert_eq!(result, Ok(()), "{domain}");
                            continue;
                        };
                        let known = engine_phase(member).filter(|name| phases.contains(name));
                        let expected = match (known, engine_limit(member)) {
                            (None, _) if phase == phases[0] => Some(DomainCause::Phase),
                            (None, _) => None,
                            (Some(declared), _) if declared != phase => None,
                            (Some(_), Err(why)) => Some(DomainCause::Malformed(why)),
                            (Some(_), Ok(EngineLimit::Relation)) => None,
                            (Some(_), Ok(_)) => match &result {
                                Err(_) => Some(DomainCause::Outside),
                                Ok(()) => None,
                            },
                        };
                        let reason = member
                            .get("reason")
                            .and_then(Value::as_str)
                            .filter(|reason| !reason.is_empty());
                        match result {
                            Ok(()) => assert_eq!(expected, None, "{domain} {config} {phase}"),
                            Err((restriction, cause)) => {
                                assert_eq!(Some(cause), expected, "{domain} {config} {phase}");
                                assert!(!restriction.is_empty(), "{domain}");
                                assert!(reason.is_none_or(|reason| reason == restriction));
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
    }

    #[test]
    fn engine_values_compare_integers_exactly() {
        let accepts = |values: Value, actual: Value| {
            check_engine_limit("x", &json!({ "values": values }), &actual).is_ok()
        };
        assert!(!accepts(
            json!([9007199254740992_u64]),
            json!(9007199254740993_u64)
        ));
        assert!(!accepts(
            json!([9007199254740993_u64]),
            json!(9007199254740992.0)
        ));
        assert!(accepts(
            json!([9007199254740992_u64]),
            json!(9007199254740992.0)
        ));
        assert!(accepts(json!([u64::MAX]), json!(u64::MAX)));
        assert!(!accepts(json!([u64::MAX]), json!(u64::MAX - 1)));
        assert!(!accepts(json!([u64::MAX]), json!(18446744073709551616.0)));
        assert!(accepts(json!([i64::MIN]), json!(i64::MIN)));
        assert!(!accepts(json!([i64::MIN]), json!(i64::MIN + 1)));
        assert!(accepts(json!([10000000]), json!(10000000.0)));
        assert!(!accepts(json!([128]), json!(128.5)));
        assert!(accepts(json!([0.5]), json!(0.5)));
        assert!(!accepts(json!([0.5]), json!(0)));
    }

    #[test]
    fn a_recipe_without_a_method_needs_a_declared_default() {
        let mut rules = json!({"semantic_rules": {},
            "quantization_recipe_classes": ["fp8", "modelopt"]});
        let config = json!({});
        let unnamed = json!({"quant_algo": "NVFP4"});
        let numbered = json!({"quant_method": 5});
        for recipe in [None, Some(&unnamed), Some(&numbered)] {
            assert!(matches!(
                admit(&config, recipe, &rules),
                Err(Refusal::RecipeClass { class: None, .. })
            ));
        }
        assert_eq!(
            admit(&config, None, &rules).unwrap_err().to_string(),
            "unsupported quantization_config.quant_method: expected one of [\"fp8\",\"modelopt\"], received absent"
        );
        rules[QUANT_METHOD_DEFAULT] = "modelopt".into();
        for recipe in [None, Some(&unnamed), Some(&numbered)] {
            assert_eq!(admit(&config, recipe, &rules), Ok("modelopt"));
        }
        let fp8 = json!({"quant_method": "fp8"});
        assert_eq!(admit(&config, Some(&fp8), &rules), Ok("fp8"));
        rules[QUANT_METHOD_DEFAULT] = "awq".into();
        assert!(matches!(
            admit(&config, None, &rules),
            Err(Refusal::RecipeClass {
                class: Some("awq"),
                method: None,
                ..
            })
        ));
    }
}
