//! deepseek-recipe's strict JSON Schema normalization, ported from recipe
//! 0.2.0 (commit 93b0fd229c): `deepseek-recipe/src/protocol/json_schema.rs`
//! lines 1-287 and `validation.rs` lines 200-264.
//!
//! The recipe deserializes a strict schema into a typed node set, normalizes
//! it and serializes it again. The port keeps its three phases over the
//! request's JSON:
//! - reading (the recipe's serde deserialization): a node is a `type` array,
//!   a single `type`, an `anyOf` list of single-type nodes or a `$ref`,
//!   chosen in that order by the field present; each type keeps only its own
//!   keywords and drops every other one. Fields are read in their arrival
//!   order and the first failure is the refusal, with the message serde
//!   writes;
//! - normalization (`JsonSchema::normalize`): every object lists all of its
//!   properties as `required` and sets `additionalProperties` to false;
//!   unknown, duplicate or partial `required` lists, `additionalProperties:
//!   true` and an object without properties below the top level are refused;
//! - writing (serde's serialization): `type` first, then the type's
//!   keywords in the recipe's declaration order, absent optional ones
//!   omitted.
//!
//! The top level must then be an object. A sequence where the recipe reads
//! one single-type node is refused with the tag serde reads first; serde's
//! positional reading of the rest is not ported.

use frontend_format::FrontendJsonV1;

/// The refusal of a value that is no schema node (the recipe's
/// `untagged_deserialize` error).
const NO_NODE: &str = "one of `type`, `anyOf`, `$ref` field is required";

/// The types a single-type node names, in the recipe's declaration order.
const KINDS: [&str; 7] = [
    "object", "string", "number", "integer", "boolean", "array", "null",
];

/// The types a `type` array lists (`SimpleJsonSchemaType`).
const SIMPLE_KINDS: [&str; 5] = ["string", "number", "integer", "boolean", "null"];

/// The `format` values a string node keeps (`StringFormat`).
const FORMATS: [&str; 5] = ["email", "hostname", "ipv4", "ipv6", "uuid"];

/// The numeric keywords a number or integer node keeps, after its
/// description, in declaration order.
const NUMERIC_KEYWORDS: [&str; 7] = [
    "const",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

/// A strict tool's parameters as the recipe stores them
/// (`normalize_tool_parameters` with `strict: true`): `parameters`
/// normalized, a missing schema being the empty object that accepts no
/// properties. An object without properties is allowed at the top level.
///
/// # Errors
///
/// The recipe's refusal of a schema it cannot normalize.
pub fn tool_parameters(parameters: Option<&FrontendJsonV1>) -> Result<FrontendJsonV1, String> {
    let default = empty_object();
    normalize(
        parameters.unwrap_or(&default),
        true,
        "Invalid tool parameters schema : ",
        "The top level of the function parameters schema must be an object.",
    )
}

/// A strict `json_schema` response format's schema normalized, as the
/// recipe checks it (`validate_response_schema` with `strict: true`); an
/// object without properties is refused at the top level too.
///
/// # Errors
///
/// The recipe's refusal of a schema it cannot normalize.
pub fn response_schema(schema: &FrontendJsonV1) -> Result<FrontendJsonV1, String> {
    normalize(
        schema,
        false,
        "Invalid json schema: ",
        "The top level of the json schema must be an object.",
    )
}

/// The recipe's default strict schema (`JsonSchema::default`).
fn empty_object() -> FrontendJsonV1 {
    FrontendJsonV1::Object(vec![
        ("type".into(), FrontendJsonV1::String("object".into())),
        ("properties".into(), FrontendJsonV1::Object(Vec::new())),
        ("additionalProperties".into(), FrontendJsonV1::Bool(false)),
        ("required".into(), FrontendJsonV1::Array(Vec::new())),
    ])
}

/// `normalize_strict_schema` (json_schema.rs lines 273-287).
fn normalize(
    schema: &FrontendJsonV1,
    allow_empty_object: bool,
    read_error: &str,
    top_level_error: &str,
) -> Result<FrontendJsonV1, String> {
    let mut node = read_node(schema).map_err(|error| format!("{read_error}{error}"))?;
    node.normalize(allow_empty_object)?;
    if !matches!(&node, Node::Type(typed) if typed.kind == "object") {
        return Err(top_level_error.to_owned());
    }
    Ok(node.write())
}

/// `JsonSchema`: one schema node.
enum Node {
    SimpleAnyOf {
        types: Vec<&'static str>,
        description: Option<String>,
    },
    Type(Typed),
    AnyOf {
        any_of: Vec<Typed>,
        description: Option<String>,
    },
    Ref(String),
}

/// `JsonSchemaType`: a node with one type and the keywords it keeps.
struct Typed {
    kind: &'static str,
    description: Option<String>,
    /// An object's properties and `$def` entries.
    properties: Option<Vec<(String, Node)>>,
    definitions: Option<Vec<(String, Node)>>,
    additional_properties: Option<bool>,
    required: Option<Vec<String>>,
    /// A string's pattern, format and enum, or a number's or integer's
    /// numeric keywords, in declaration order.
    keywords: Vec<(&'static str, FrontendJsonV1)>,
    /// An array's items.
    items: Option<Box<Node>>,
}

/// The recipe's untagged `JsonSchema` reading (json_schema.rs lines 13-40):
/// the first variant whose field is present, its fields read in
/// declaration order, a failure naming the field.
fn read_node(value: &FrontendJsonV1) -> Result<Node, String> {
    let FrontendJsonV1::Object(entries) = value else {
        return Err(NO_NODE.into());
    };
    let field = |name: &str| {
        entries
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    };
    let named = |name: &str| field(name).unwrap_or(&FrontendJsonV1::Null);
    let labeled = |name: &str, error: String| format!("field `{name}`: {error}");
    let description =
        |name: &str| optional(named(name), read_string).map_err(|error| labeled(name, error));
    match field("type") {
        Some(FrontendJsonV1::Array(_)) => Ok(Node::SimpleAnyOf {
            types: sequence(named("type"), |value| variant(value, &SIMPLE_KINDS))
                .map_err(|error| labeled("type", error))?,
            description: description("description")?,
        }),
        Some(_) => read_typed(value).map(Node::Type),
        None if field("anyOf").is_some() => Ok(Node::AnyOf {
            any_of: sequence(named("anyOf"), read_typed)
                .map_err(|error| labeled("anyOf", error))?,
            description: description("description")?,
        }),
        None if field("$ref").is_some() => Ok(Node::Ref(
            read_string(named("$ref")).map_err(|error| labeled("$ref", error))?,
        )),
        None => Err(NO_NODE.into()),
    }
}

/// `JsonSchemaType`, internally tagged by `type`: the tag is read first,
/// then the type's own fields in their arrival order, every other key
/// ignored.
fn read_typed(value: &FrontendJsonV1) -> Result<Typed, String> {
    let entries = match value {
        FrontendJsonV1::Object(entries) => entries,
        FrontendJsonV1::Array(values) => {
            let Some(tag) = values.first() else {
                return Err("missing field `type`".into());
            };
            tag_kind(tag)?;
            return Err(invalid_type(value, "internally tagged enum JsonSchemaType"));
        }
        other => return Err(invalid_type(other, "internally tagged enum JsonSchemaType")),
    };
    let tag = entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
        .ok_or("missing field `type`")?;
    let mut typed = Typed {
        kind: tag_kind(tag)?,
        description: None,
        properties: None,
        definitions: None,
        additional_properties: None,
        required: None,
        keywords: Vec::new(),
        items: None,
    };
    let mut items = None;
    for (key, value) in entries {
        let key = key.as_str();
        match (typed.kind, key) {
            (_, "description") => typed.description = optional(value, read_string)?,
            ("object", "properties") => typed.properties = optional(value, read_nodes)?,
            ("object", "$def") => typed.definitions = optional(value, read_nodes)?,
            ("object", "additionalProperties") => {
                typed.additional_properties = optional(value, read_bool)?;
            }
            ("object", "required") => {
                typed.required = optional(value, |value| sequence(value, read_string))?;
            }
            // Not an `Option`: a null is read as a node.
            ("array", "items") => items = Some(read_node(value)?),
            // A null keyword is absent.
            (_, _) if matches!(value, FrontendJsonV1::Null) => {}
            ("string", "pattern" | "format" | "enum") => {
                let kept = match key {
                    "pattern" => FrontendJsonV1::String(read_string(value)?),
                    "format" => FrontendJsonV1::String(variant(value, &FORMATS)?.into()),
                    _ => {
                        sequence(value, read_string)?;
                        value.clone()
                    }
                };
                typed.keywords.push((keyword(key), kept));
            }
            ("number" | "integer", _) if NUMERIC_KEYWORDS.contains(&key) => {
                if typed.kind == "number" {
                    read_number(value)?;
                } else {
                    read_i64(value)?;
                }
                typed.keywords.push((keyword(key), value.clone()));
            }
            _ => {}
        }
    }
    if typed.kind == "array" {
        typed.items = Some(Box::new(items.ok_or("missing field `items`")?));
    }
    // Pattern, format and enum, and the numeric keywords, are written in
    // declaration order whatever order they arrived in.
    let order = |name: &str| {
        ["pattern", "format", "enum"]
            .iter()
            .chain(&NUMERIC_KEYWORDS)
            .position(|keyword| *keyword == name)
    };
    typed.keywords.sort_by_key(|(name, _)| order(name));
    Ok(typed)
}

/// The static spelling of a kept keyword.
fn keyword(name: &str) -> &'static str {
    ["pattern", "format", "enum"]
        .iter()
        .chain(&NUMERIC_KEYWORDS)
        .find(|keyword| **keyword == name)
        .expect("a kept keyword")
}

/// A type tag: one of [`KINDS`].
fn tag_kind(tag: &FrontendJsonV1) -> Result<&'static str, String> {
    match tag {
        FrontendJsonV1::String(name) => KINDS
            .iter()
            .find(|kind| **kind == name.as_str())
            .copied()
            .ok_or_else(|| unknown_variant(name, &KINDS)),
        other => Err(invalid_type(other, "variant identifier")),
    }
}

/// A unit enum value (`SimpleJsonSchemaType`, `StringFormat`): a variant
/// name, or a map of one variant name to null, as serde reads an
/// externally tagged enum.
fn variant(value: &FrontendJsonV1, variants: &[&'static str]) -> Result<&'static str, String> {
    let known = |name: &str| {
        variants
            .iter()
            .find(|variant| **variant == name)
            .copied()
            .ok_or_else(|| unknown_variant(name, variants))
    };
    match value {
        FrontendJsonV1::String(name) => known(name),
        FrontendJsonV1::Object(entries) => match entries.as_slice() {
            [(name, unit)] => {
                let name = known(name)?;
                match unit {
                    FrontendJsonV1::Null => Ok(name),
                    other => Err(invalid_type(other, "unit")),
                }
            }
            _ => Err("invalid value: map, expected map with a single key".into()),
        },
        other => Err(invalid_type(other, "string or map")),
    }
}

fn unknown_variant(name: &str, variants: &[&str]) -> String {
    let expected: Vec<String> = variants
        .iter()
        .map(|variant| format!("`{variant}`"))
        .collect();
    format!(
        "unknown variant `{name}`, expected one of {}",
        expected.join(", ")
    )
}

/// An `Option` field: null is absent.
fn optional<T>(
    value: &FrontendJsonV1,
    read: impl FnOnce(&FrontendJsonV1) -> Result<T, String>,
) -> Result<Option<T>, String> {
    match value {
        FrontendJsonV1::Null => Ok(None),
        value => read(value).map(Some),
    }
}

fn sequence<T>(
    value: &FrontendJsonV1,
    read: impl Fn(&FrontendJsonV1) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    match value {
        FrontendJsonV1::Array(values) => values.iter().map(read).collect(),
        other => Err(invalid_type(other, "a sequence")),
    }
}

/// A map of schema nodes (`IndexMap<String, Box<JsonSchema>>`).
fn read_nodes(value: &FrontendJsonV1) -> Result<Vec<(String, Node)>, String> {
    match value {
        FrontendJsonV1::Object(entries) => entries
            .iter()
            .map(|(key, value)| Ok((key.clone(), read_node(value)?)))
            .collect(),
        other => Err(invalid_type(other, "a map")),
    }
}

fn read_string(value: &FrontendJsonV1) -> Result<String, String> {
    match value {
        FrontendJsonV1::String(text) => Ok(text.clone()),
        other => Err(invalid_type(other, "a string")),
    }
}

fn read_bool(value: &FrontendJsonV1) -> Result<bool, String> {
    match value {
        FrontendJsonV1::Bool(flag) => Ok(*flag),
        other => Err(invalid_type(other, "a boolean")),
    }
}

fn read_number(value: &FrontendJsonV1) -> Result<(), String> {
    match value {
        FrontendJsonV1::Number(_) => Ok(()),
        other => Err(invalid_type(other, "a JSON number")),
    }
}

/// An `i64`: an integer in range; a larger one is an invalid value and a
/// fraction an invalid type, as serde_json reads them.
fn read_i64(value: &FrontendJsonV1) -> Result<(), String> {
    match value {
        FrontendJsonV1::Number(number) => match number_kind(number) {
            NumberKind::Signed(_) => Ok(()),
            NumberKind::Unsigned(unsigned) => {
                Err(format!("invalid value: integer `{unsigned}`, expected i64"))
            }
            NumberKind::Float(_) => Err(invalid_type(value, "i64")),
        },
        other => Err(invalid_type(other, "i64")),
    }
}

/// How serde_json holds a number: an `i64`, else a `u64`, else an `f64`;
/// a fraction or an exponent is always an `f64`.
enum NumberKind {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

fn number_kind(number: &serde_json::Number) -> NumberKind {
    let text = number.to_string();
    if !text.contains(['.', 'e', 'E']) {
        if let Ok(signed) = text.parse() {
            return NumberKind::Signed(signed);
        }
        if let Ok(unsigned) = text.parse() {
            return NumberKind::Unsigned(unsigned);
        }
    }
    NumberKind::Float(text.parse().unwrap_or(f64::NAN))
}

/// serde's `invalid type` refusal, the value spelled as serde_json's
/// `Unexpected` writes it.
fn invalid_type(value: &FrontendJsonV1, expected: &str) -> String {
    let unexpected = match value {
        FrontendJsonV1::Null => "null".to_owned(),
        FrontendJsonV1::Bool(flag) => format!("boolean `{flag}`"),
        FrontendJsonV1::Number(number) => match number_kind(number) {
            NumberKind::Signed(signed) => format!("integer `{signed}`"),
            NumberKind::Unsigned(unsigned) => format!("integer `{unsigned}`"),
            NumberKind::Float(float) => format!("floating point `{}`", float_text(float)),
        },
        FrontendJsonV1::String(text) => format!("string {text:?}"),
        FrontendJsonV1::Array(_) => "sequence".to_owned(),
        FrontendJsonV1::Object(_) => "map".to_owned(),
    };
    format!("invalid type: {unexpected}, expected {expected}")
}

/// A float as the recipe's serde_json writes it in a refusal: the shortest
/// digits that read back, with a signed exponent below 1e-4 and from 1e16.
fn float_text(float: f64) -> String {
    let text = format!("{float:?}");
    match text.split_once('e') {
        Some((mantissa, exponent)) if !exponent.starts_with('-') => {
            format!("{mantissa}e+{exponent}")
        }
        _ => text,
    }
}

impl Node {
    /// `JsonSchema::normalize` (json_schema.rs lines 158-166).
    fn normalize(&mut self, allow_empty_object: bool) -> Result<(), String> {
        match self {
            Node::Type(typed) => typed.normalize(allow_empty_object),
            Node::AnyOf { any_of, .. } => any_of
                .iter_mut()
                .try_for_each(|item| item.normalize(allow_empty_object)),
            Node::SimpleAnyOf { .. } | Node::Ref(_) => Ok(()),
        }
    }

    /// The node as serde writes it.
    fn write(&self) -> FrontendJsonV1 {
        let mut entries = Vec::new();
        let description = |entries: &mut Vec<(String, FrontendJsonV1)>, text: &Option<String>| {
            if let Some(text) = text {
                entries.push(("description".into(), FrontendJsonV1::String(text.clone())));
            }
        };
        match self {
            Node::SimpleAnyOf {
                types,
                description: text,
            } => {
                let types = types
                    .iter()
                    .map(|kind| FrontendJsonV1::String((*kind).into()))
                    .collect();
                entries.push(("type".into(), FrontendJsonV1::Array(types)));
                description(&mut entries, text);
            }
            Node::Type(typed) => return typed.write(),
            Node::AnyOf {
                any_of,
                description: text,
            } => {
                let any_of = any_of.iter().map(Typed::write).collect();
                entries.push(("anyOf".into(), FrontendJsonV1::Array(any_of)));
                description(&mut entries, text);
            }
            Node::Ref(reference) => {
                entries.push(("$ref".into(), FrontendJsonV1::String(reference.clone())));
            }
        }
        FrontendJsonV1::Object(entries)
    }
}

impl Typed {
    /// `JsonSchemaType::normalize` (json_schema.rs lines 188-253).
    fn normalize(&mut self, allow_empty_object: bool) -> Result<(), String> {
        match self.kind {
            "object" => {
                let mut properties = self.properties.take().unwrap_or_default();
                if !allow_empty_object && properties.is_empty() {
                    return Err("An object with no properties is not allowed.".into());
                }
                for (_, value) in &mut properties {
                    value.normalize(false)?;
                }
                for (_, value) in self.definitions.iter_mut().flatten() {
                    value.normalize(false)?;
                }
                if let Some(required) = &self.required {
                    let mut seen = std::collections::BTreeSet::new();
                    for key in required {
                        if !properties.iter().any(|(name, _)| name == key) {
                            return Err(format!("Required property '{key}' not found in object"));
                        }
                        if !seen.insert(key) {
                            return Err(format!("Duplicate required property '{key}'"));
                        }
                    }
                    if seen.len() != properties.len() {
                        return Err(
                            "Required properties must match all properties in the object".into(),
                        );
                    }
                }
                if self.additional_properties == Some(true) {
                    return Err("additionalProperties should not be true".into());
                }
                self.required = Some(properties.iter().map(|(name, _)| name.clone()).collect());
                self.additional_properties = Some(false);
                self.properties = Some(properties);
            }
            "array" => {
                if let Some(items) = &mut self.items {
                    items.normalize(false)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The node as serde writes it: the tag, then the fields in
    /// declaration order, an absent optional one omitted. An object's
    /// properties, `additionalProperties` and `required` are always
    /// written.
    fn write(&self) -> FrontendJsonV1 {
        let mut entries = vec![("type".to_owned(), FrontendJsonV1::String(self.kind.into()))];
        let nodes = |nodes: &Vec<(String, Node)>| {
            FrontendJsonV1::Object(
                nodes
                    .iter()
                    .map(|(key, node)| (key.clone(), node.write()))
                    .collect(),
            )
        };
        let description = self.description.as_ref().map(|text| {
            (
                "description".to_owned(),
                FrontendJsonV1::String(text.clone()),
            )
        });
        match self.kind {
            "object" => {
                entries.push((
                    "properties".into(),
                    self.properties.as_ref().map_or(FrontendJsonV1::Null, nodes),
                ));
                entries.push((
                    "additionalProperties".into(),
                    self.additional_properties
                        .map_or(FrontendJsonV1::Null, FrontendJsonV1::Bool),
                ));
                entries.push((
                    "required".into(),
                    self.required
                        .as_ref()
                        .map_or(FrontendJsonV1::Null, |names| {
                            FrontendJsonV1::Array(
                                names
                                    .iter()
                                    .map(|name| FrontendJsonV1::String(name.clone()))
                                    .collect(),
                            )
                        }),
                ));
                entries.extend(description);
                if let Some(definitions) = &self.definitions {
                    entries.push(("$def".into(), nodes(definitions)));
                }
            }
            "array" => {
                if let Some(items) = &self.items {
                    entries.push(("items".into(), items.write()));
                }
                entries.extend(description);
            }
            _ => {
                entries.extend(description);
                entries.extend(
                    self.keywords
                        .iter()
                        .map(|(name, value)| ((*name).to_owned(), value.clone())),
                );
            }
        }
        FrontendJsonV1::Object(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(value: serde_json::Value) -> FrontendJsonV1 {
        serde_json::from_value(value).unwrap()
    }

    fn text(value: &FrontendJsonV1) -> String {
        value.to_json_value().to_string()
    }

    /// The recipe's own test helper: a tool's normalization, an empty
    /// object allowed at the top level.
    fn normalized(schema: serde_json::Value) -> Result<serde_json::Value, String> {
        normalize(&json(schema), true, "", "top level").map(|value| value.to_json_value())
    }

    fn refused(schema: serde_json::Value) -> String {
        normalized(schema).expect_err("the schema is refused")
    }

    /// json_schema.rs lines 300-317.
    #[test]
    fn normalize_fills_required_and_additional_properties() {
        let normalized = normalized(serde_json::json!({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }))
        .unwrap();
        assert_eq!(normalized["required"], serde_json::json!(["b", "a"]));
        assert_eq!(normalized["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            normalized["properties"]["a"]["required"],
            serde_json::json!(["x"])
        );
        assert_eq!(
            normalized["properties"]["a"]["additionalProperties"],
            serde_json::json!(false)
        );
    }

    /// json_schema.rs lines 319-347.
    #[test]
    fn normalize_rejects_inconsistent_required_lists() {
        let with = |required: serde_json::Value| {
            refused(serde_json::json!({
                "type": "object", "properties": {"a": {"type": "string"}}, "required": required}))
        };
        assert_eq!(
            with(serde_json::json!(["z"])),
            "Required property 'z' not found in object"
        );
        assert_eq!(
            with(serde_json::json!(["a", "a"])),
            "Duplicate required property 'a'"
        );
        assert_eq!(
            refused(serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            })),
            "Required properties must match all properties in the object"
        );
        assert_eq!(
            refused(serde_json::json!({
                "type": "object", "properties": {"a": {"type": "string"}},
                "additionalProperties": true})),
            "additionalProperties should not be true"
        );
    }

    /// json_schema.rs lines 349-365.
    #[test]
    fn normalize_rejects_empty_nested_objects_and_non_object_top_levels() {
        assert_eq!(
            refused(serde_json::json!({"type": "object", "properties": {"a": {"type": "object"}}})),
            "An object with no properties is not allowed."
        );
        assert_eq!(refused(serde_json::json!({"type": "string"})), "top level");
        assert!(refused(serde_json::json!({"description": "no type"})).contains(NO_NODE));
    }

    /// A tool's parameters (validation.rs lines 209-225): a missing schema
    /// is the empty object, which an empty top-level object also becomes;
    /// a response schema refuses an empty top-level object.
    #[test]
    fn tools_allow_an_empty_top_level_object_and_response_schemas_do_not() {
        let empty =
            r#"{"type":"object","properties":{},"additionalProperties":false,"required":[]}"#;
        assert_eq!(text(&tool_parameters(None).unwrap()), empty);
        let object = json(serde_json::json!({"type": "object"}));
        assert_eq!(text(&tool_parameters(Some(&object)).unwrap()), empty);
        assert_eq!(
            response_schema(&object).unwrap_err(),
            "An object with no properties is not allowed."
        );
        for (schema, tool, response) in [
            (
                serde_json::json!({}),
                format!("Invalid tool parameters schema : {NO_NODE}"),
                format!("Invalid json schema: {NO_NODE}"),
            ),
            (
                serde_json::json!(false),
                format!("Invalid tool parameters schema : {NO_NODE}"),
                format!("Invalid json schema: {NO_NODE}"),
            ),
            (
                serde_json::json!({"type": "string"}),
                "The top level of the function parameters schema must be an object.".into(),
                "The top level of the json schema must be an object.".into(),
            ),
        ] {
            let schema = json(schema);
            assert_eq!(tool_parameters(Some(&schema)).unwrap_err(), tool);
            assert_eq!(response_schema(&schema).unwrap_err(), response);
        }
    }

    /// Each type keeps its own keywords in declaration order and drops the
    /// rest, as deepseek-recipe 0.2.0 normalizes them.
    #[test]
    fn each_type_keeps_its_keywords_in_declaration_order() {
        let property = |schema: serde_json::Value| {
            let normalized =
                normalized(serde_json::json!({"type": "object", "properties": {"a": schema}}))
                    .unwrap();
            normalized["properties"]["a"].to_string()
        };
        for (schema, expected) in [
            (
                serde_json::json!({"enum": ["x"], "pattern": "p", "type": "string", "format": "uuid",
                    "description": "d", "minLength": 2}),
                r#"{"type":"string","description":"d","pattern":"p","format":"uuid","enum":["x"]}"#,
            ),
            (
                serde_json::json!({"type": "string", "format": {"email": null}, "description": null}),
                r#"{"type":"string","format":"email"}"#,
            ),
            (
                serde_json::json!({"type": "integer", "multipleOf": 2, "exclusiveMaximum": 9,
                    "const": 4, "default": 2, "minimum": -1, "maximum": 10,
                    "exclusiveMinimum": -2, "description": "d"}),
                r#"{"type":"integer","description":"d","const":4,"default":2,"minimum":-1,"maximum":10,"exclusiveMinimum":-2,"exclusiveMaximum":9,"multipleOf":2}"#,
            ),
            (
                serde_json::json!({"type": "number", "maximum": 1e2, "minimum": 1.5, "default": 3,
                    "description": "d", "multipleOf": null}),
                r#"{"type":"number","description":"d","default":3,"minimum":1.5,"maximum":100.0}"#,
            ),
            (
                serde_json::json!({"type": "boolean", "default": true, "description": "d"}),
                r#"{"type":"boolean","description":"d"}"#,
            ),
            (serde_json::json!({"type": "null"}), r#"{"type":"null"}"#),
            (
                serde_json::json!({"type": ["string", "null"], "enum": ["x"], "description": "d"}),
                r#"{"type":["string","null"],"description":"d"}"#,
            ),
            (
                serde_json::json!({"type": [{"string": null}]}),
                r#"{"type":["string"]}"#,
            ),
            (
                serde_json::json!({"description": "d", "items": {"type": "object",
                    "properties": {"x": {"type": "boolean"}}}, "type": "array", "minItems": 1}),
                r#"{"type":"array","items":{"type":"object","properties":{"x":{"type":"boolean"}},"additionalProperties":false,"required":["x"]},"description":"d"}"#,
            ),
            (
                serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null", "description": "n"}],
                    "description": "d", "title": "t"}),
                r#"{"anyOf":[{"type":"string"},{"type":"null","description":"n"}],"description":"d"}"#,
            ),
            (
                serde_json::json!({"type": "string", "anyOf": [{"type": "integer"}]}),
                r#"{"type":"string"}"#,
            ),
            (
                serde_json::json!({"$ref": "#/x", "description": 5}),
                r##"{"$ref":"#/x"}"##,
            ),
        ] {
            assert_eq!(property(schema.clone()), expected, "{schema}");
        }
        // An object keeps its description and `$def`, normalized, and drops
        // `$defs` and other keywords.
        let normalized = normalized(serde_json::json!({
            "$defs": {"z": {"type": "string"}}, "description": "top", "title": "T",
            "$def": {"x": {"type": "object", "properties": {"y": {"type": "string"}}}},
            "properties": {"a": {"$ref": "#/$def/x"}}, "type": "object",
        }))
        .unwrap();
        assert_eq!(
            normalized.to_string(),
            r##"{"type":"object","properties":{"a":{"$ref":"#/$def/x"}},"additionalProperties":false,"required":["a"],"description":"top","$def":{"x":{"type":"object","properties":{"y":{"type":"string"}},"additionalProperties":false,"required":["y"]}}}"##
        );
        // A null `properties`, `required` or `additionalProperties` is absent.
        assert_eq!(
            normalized_text(serde_json::json!({"type": "object", "properties": null})),
            r#"{"type":"object","properties":{},"additionalProperties":false,"required":[]}"#
        );
    }

    fn normalized_text(schema: serde_json::Value) -> String {
        normalized(schema).unwrap().to_string()
    }

    /// The refusals deepseek-recipe 0.2.0 gives for schemas it cannot read,
    /// with serde's messages: the first failure in arrival order, the type
    /// tag first, and every reading failure before any normalization one.
    #[test]
    fn reading_refusals_are_the_recipes() {
        let property = |schema: serde_json::Value| {
            refused(serde_json::json!({"type": "object", "properties": {"a": schema}}))
        };
        let kinds = "`object`, `string`, `number`, `integer`, `boolean`, `array`, `null`";
        let simple = "`string`, `number`, `integer`, `boolean`, `null`";
        for (schema, expected) in [
            (serde_json::json!({"description": "x"}), NO_NODE.to_owned()),
            (
                serde_json::json!({"type": "date"}),
                format!("unknown variant `date`, expected one of {kinds}"),
            ),
            (
                serde_json::json!({"type": 5}),
                "invalid type: integer `5`, expected variant identifier".into(),
            ),
            (
                serde_json::json!({"type": null}),
                "invalid type: null, expected variant identifier".into(),
            ),
            (
                serde_json::json!({"type": ["object"]}),
                format!("field `type`: unknown variant `object`, expected one of {simple}"),
            ),
            (
                serde_json::json!({"type": [1]}),
                "field `type`: invalid type: integer `1`, expected string or map".into(),
            ),
            (
                serde_json::json!({"type": [{}]}),
                "field `type`: invalid value: map, expected map with a single key".into(),
            ),
            (
                serde_json::json!({"type": ["string"], "description": 5}),
                "field `description`: invalid type: integer `5`, expected a string".into(),
            ),
            (
                serde_json::json!({"type": "string", "description": 5}),
                "invalid type: integer `5`, expected a string".into(),
            ),
            (
                serde_json::json!({"type": "string", "format": "date-time"}),
                "unknown variant `date-time`, expected one of `email`, `hostname`, `ipv4`, \
                 `ipv6`, `uuid`"
                    .into(),
            ),
            (
                serde_json::json!({"type": "string", "format": {"email": 1}}),
                "invalid type: integer `1`, expected unit".into(),
            ),
            (
                serde_json::json!({"type": "string", "format": ["email"]}),
                "invalid type: sequence, expected string or map".into(),
            ),
            (
                serde_json::json!({"type": "string", "enum": ["a", 1]}),
                "invalid type: integer `1`, expected a string".into(),
            ),
            (
                serde_json::json!({"type": "string", "enum": [null]}),
                "invalid type: null, expected a string".into(),
            ),
            (
                serde_json::json!({"type": "string", "enum": "a"}),
                "invalid type: string \"a\", expected a sequence".into(),
            ),
            (
                serde_json::json!({"type": "integer", "minimum": 1.5}),
                "invalid type: floating point `1.5`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "integer", "minimum": 1.0}),
                "invalid type: floating point `1.0`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "integer", "minimum": 1e300}),
                "invalid type: floating point `1e+300`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "integer", "minimum": 1e-7}),
                "invalid type: floating point `1e-7`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "integer", "maximum": 9223372036854775808_u64}),
                "invalid value: integer `9223372036854775808`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "integer", "minimum": true}),
                "invalid type: boolean `true`, expected i64".into(),
            ),
            (
                serde_json::json!({"type": "number", "minimum": "1"}),
                "invalid type: string \"1\", expected a JSON number".into(),
            ),
            (
                serde_json::json!({"type": "array"}),
                "missing field `items`".into(),
            ),
            (
                serde_json::json!({"type": "array", "items": null}),
                NO_NODE.into(),
            ),
            (
                serde_json::json!({"anyOf": [{"$ref": "#/x"}]}),
                "field `anyOf`: missing field `type`".into(),
            ),
            (
                serde_json::json!({"anyOf": [5]}),
                "field `anyOf`: invalid type: integer `5`, expected internally tagged enum \
                 JsonSchemaType"
                    .into(),
            ),
            (
                serde_json::json!({"anyOf": [{"type": ["string"]}]}),
                "field `anyOf`: invalid type: sequence, expected variant identifier".into(),
            ),
            (
                serde_json::json!({"anyOf": null}),
                "field `anyOf`: invalid type: null, expected a sequence".into(),
            ),
            (
                serde_json::json!({"anyOf": {"type": "string"}}),
                "field `anyOf`: invalid type: map, expected a sequence".into(),
            ),
            (
                serde_json::json!({"$ref": 5}),
                "field `$ref`: invalid type: integer `5`, expected a string".into(),
            ),
        ] {
            assert_eq!(property(schema.clone()), expected, "{schema}");
        }
        for (schema, expected) in [
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}},
                    "additionalProperties": {}}),
                "invalid type: map, expected a boolean",
            ),
            (
                serde_json::json!({"type": "object", "properties": []}),
                "invalid type: sequence, expected a map",
            ),
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}},
                    "required": [1]}),
                "invalid type: integer `1`, expected a string",
            ),
            // The first failing field in arrival order.
            (
                serde_json::json!({"type": "object", "additionalProperties": {},
                    "properties": {"a": {"type": "date"}}}),
                "invalid type: map, expected a boolean",
            ),
            // Reading fails before normalization does.
            (
                serde_json::json!({"type": "object",
                    "properties": {"a": {"type": "object"}, "b": {"type": "x"}}}),
                "unknown variant `x`, expected one of `object`, `string`, `number`, \
                 `integer`, `boolean`, `array`, `null`",
            ),
        ] {
            assert_eq!(refused(schema.clone()), expected, "{schema}");
        }
    }

    /// Normalization refuses in the recipe's order: properties, then
    /// `$def`, then `required`, then `additionalProperties`; nested empty
    /// objects are refused under arrays and `anyOf` too.
    #[test]
    fn normalization_refusals_follow_the_recipes_order() {
        let empty = "An object with no properties is not allowed.";
        for (schema, expected) in [
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "object"}},
                    "required": ["z"], "additionalProperties": true}),
                empty,
            ),
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}},
                    "required": ["a", "z"], "additionalProperties": true}),
                "Required property 'z' not found in object",
            ),
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}},
                    "$def": {"x": {"type": "object"}}}),
                empty,
            ),
            (
                serde_json::json!({"type": "object", "properties": {"a": {"type": "array",
                    "items": {"type": "object"}}}}),
                empty,
            ),
            (
                serde_json::json!({"type": "object", "properties": {"a": {"anyOf": [
                    {"type": "object"}]}}}),
                empty,
            ),
            (
                serde_json::json!({"anyOf": [{"type": "object", "properties": {
                    "a": {"type": "string"}}}]}),
                "top level",
            ),
        ] {
            assert_eq!(refused(schema.clone()), expected, "{schema}");
        }
        // A complete required list and a false additionalProperties stand.
        assert_eq!(
            normalized_text(serde_json::json!({"type": "object",
                "properties": {"a": {"type": "string"}}, "additionalProperties": false,
                "required": ["a"]})),
            r#"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false,"required":["a"]}"#
        );
    }
}
