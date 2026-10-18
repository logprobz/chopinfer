//! JSON and the supported JSON Schema subset, as grammar expressions.
//!
//! The text rules follow xgrammar's defaults as the reference serving stack
//! uses them for free generation: properties in schema order, a schema with
//! `properties` and no
//! `additionalProperties` admits no other keys. Whitespace runs are bounded
//! by default; callers can select unbounded or compact reference layout.
//!
//! Unsupported assertions and combinations fail before generation. Keywords
//! beside `const` and `enum` narrow the literal values; an empty intersection
//! is refused instead of restoring the original literals. Container literals
//! are validated by their keys and values, preserving their original key order.

use std::collections::HashMap;

use serde_json::Value;

use serde_json::Map;

use crate::automaton::Automaton;
use crate::budget::Budget;
use crate::expr::{CharClass, Expr, Grammar, GrammarBuilder, GrammarError, RuleId};
use crate::range::{Bound, integer_range, number_range, unsigned_range};
use crate::regex::{digits, regex_expr};
use crate::stack::Config;

/// How JSON text may be laid out.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct JsonOptions {
    /// Flexible whitespace between tokens; otherwise compact separators.
    pub any_whitespace: bool,
    /// Maximum characters per whitespace run; `None` restores xgrammar's
    /// unbounded default. Ignored with compact separators.
    pub max_whitespace_cnt: Option<u32>,
}

impl Default for JsonOptions {
    fn default() -> Self {
        Self {
            any_whitespace: true,
            max_whitespace_cnt: Some(257),
        }
    }
}

const MAX_LENGTH_BOUND: u64 = 1 << 12;
const MAX_ITEMS_BOUND: u64 = 1 << 10;

const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
    "strict",
    "discriminator",
];

/// A grammar for any JSON object.
#[must_use]
pub fn json_object_grammar(options: JsonOptions) -> Grammar {
    let mut builder = GrammarBuilder::new();
    let mut compiler = SchemaCompiler::new(&mut builder, &Value::Null, options);
    let object = compiler.any_object();
    let root = builder.rule(object);
    builder.finish(root).expect("every rule is defined")
}

/// A grammar for JSON text valid under `schema`.
pub fn json_schema_grammar(schema: &Value, options: JsonOptions) -> Result<Grammar, GrammarError> {
    let mut builder = GrammarBuilder::new();
    let expr = SchemaCompiler::new(&mut builder, schema, options).schema(schema)?;
    let root = builder.rule(expr);
    builder.finish(root)
}

/// Compiles schemas into rules of a grammar a caller is building, so a
/// frontend can embed a function's argument schema in a larger structure.
pub struct SchemaCompiler<'a> {
    builder: &'a mut GrammarBuilder,
    document: &'a Value,
    options: JsonOptions,
    any_value: Option<RuleId>,
    references: HashMap<String, RuleId>,
    depth: usize,
    budget: Budget,
    /// Whether a number with a fraction or exponent stays a finite binary64
    /// ([`Self::finite_numbers`]).
    finite_numbers: bool,
}

impl<'a> SchemaCompiler<'a> {
    /// `document` is the schema `$ref` pointers resolve against.
    pub fn new(builder: &'a mut GrammarBuilder, document: &'a Value, options: JsonOptions) -> Self {
        Self {
            builder,
            document,
            options,
            any_value: None,
            references: HashMap::new(),
            depth: 0,
            budget: Budget::default(),
            finite_numbers: false,
        }
    }

    /// Admits a number with a fraction or exponent only when it is a finite
    /// binary64, at any depth, for a reader that keeps any other such
    /// number as text. An integer keeps every digit. Magnitudes from 1e308,
    /// just below the largest finite one, are refused.
    #[must_use]
    pub fn finite_numbers(mut self) -> Self {
        self.finite_numbers = true;
        self
    }

    fn number_expr(&self) -> Expr {
        if self.finite_numbers {
            finite_number_expr()
        } else {
            number_expr()
        }
    }

    fn number(&self, fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
        let (lower, upper) = numeric_bounds(fields, false)?;
        if lower.is_none() && upper.is_none() {
            return Ok(self.number_expr());
        }
        Ok(number_range(
            lower,
            upper,
            self.finite_numbers.then_some(FINITE_INTEGER_DIGITS),
        ))
    }

    /// Cumulative construction costs, including nested literal validation.
    #[must_use]
    pub fn compilation_usage(&self) -> crate::CompileUsage {
        self.budget.usage()
    }

    fn whitespace(&self) -> Expr {
        if self.options.any_whitespace {
            // Keep the reference alphabet in compatibility mode. Bounded
            // layout also admits CR for JSON formatted with CRLF newlines.
            let alphabet = if self.options.max_whitespace_cnt.is_some() {
                " \n\t\r"
            } else {
                " \n\t"
            };
            Expr::Class(CharClass::of(alphabet)).repeat(0, self.options.max_whitespace_cnt)
        } else {
            Expr::Empty
        }
    }

    fn separator(&self, mark: &str) -> Expr {
        if self.options.any_whitespace {
            Expr::seq([self.whitespace(), Expr::literal(mark), self.whitespace()])
        } else {
            Expr::literal(&format!("{mark} "))
        }
    }

    fn comma(&self) -> Expr {
        // Each member/item owns its trailing whitespace, including before
        // a comma. Empty containers then have exactly one whitespace run.
        if self.options.any_whitespace {
            Expr::seq([Expr::literal(","), self.whitespace()])
        } else {
            Expr::literal(", ")
        }
    }

    fn colon(&self) -> Expr {
        self.separator(":")
    }

    /// Any JSON value, through one shared recursive rule.
    pub fn any_value(&mut self) -> Expr {
        let rule = match self.any_value {
            Some(rule) => rule,
            None => {
                let rule = self.builder.declare();
                self.any_value = Some(rule);
                let value = Expr::Rule(rule);
                let object = self.object_of(Vec::new(), Some(value.clone()));
                let array = self.array_of(&[], Some(value), 0, None);
                let scalar = Expr::alt([
                    string_expr(),
                    self.number_expr(),
                    Expr::literal("true"),
                    Expr::literal("false"),
                    Expr::literal("null"),
                ]);
                self.builder
                    .define(rule, Expr::alt([object, array, scalar]));
                rule
            }
        };
        Expr::Rule(rule)
    }

    /// Any JSON object.
    pub fn any_object(&mut self) -> Expr {
        let value = self.any_value();
        self.object_of(Vec::new(), Some(value))
    }

    /// Text valid under `schema`.
    pub fn schema(&mut self, schema: &Value) -> Result<Expr, GrammarError> {
        let _depth = self.budget.enter_schema()?;
        if self.depth == 0 {
            self.budget.schema(schema)?;
        }
        self.budget.charge(1, 128)?;
        self.depth += 1;
        let result = self.schema_inner(schema);
        self.depth -= 1;
        if self.depth == 0 && result.is_ok() {
            self.budget.report("schema", schema);
        }
        result
    }

    fn schema_inner(&mut self, schema: &Value) -> Result<Expr, GrammarError> {
        let fields = match schema {
            Value::Bool(true) => return Ok(self.any_value()),
            Value::Bool(false) => return Ok(Expr::Nothing),
            Value::Object(fields) => fields,
            _ => {
                return Err(GrammarError::invalid(
                    "a JSON schema must be an object or a boolean",
                ));
            }
        };
        self.budget.schema(schema)?;
        let applicable = applicable_fields(fields);
        let fields = &applicable;
        const SUPPORTED: &[&str] = &[
            "$ref",
            "const",
            "enum",
            "anyOf",
            "oneOf",
            "allOf",
            "type",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "items",
            "prefixItems",
            "additionalItems",
            "minItems",
            "maxItems",
            "properties",
            "required",
            "additionalProperties",
        ];
        for keyword in constraints(fields) {
            if !SUPPORTED.contains(&keyword) {
                return Err(GrammarError::invalid(format!(
                    "unsupported schema keyword '{keyword}'"
                )));
            }
        }
        for primary in ["$ref", "anyOf", "oneOf", "allOf"] {
            if fields.contains_key(primary)
                && let Some(sibling) = constraints(fields).find(|key| *key != primary)
            {
                return Err(GrammarError::invalid(format!(
                    "schema keyword '{sibling}' beside '{primary}' is not supported"
                )));
            }
        }
        let integer_numbers_only = match fields.get("type") {
            Some(Value::String(kind)) => kind == "integer",
            Some(Value::Array(kinds)) => {
                kinds.iter().any(|kind| kind == "integer")
                    && !kinds.iter().any(|kind| kind == "number")
            }
            _ => false,
        };
        if fields.contains_key("multipleOf") && !integer_numbers_only {
            return Err(GrammarError::invalid("multipleOf requires type integer"));
        }
        if let Some(reference) = fields.get("$ref") {
            let Value::String(reference) = reference else {
                return Err(GrammarError::invalid("$ref must be a string"));
            };
            return self.reference(reference);
        }
        let literals = match (fields.get("const"), fields.get("enum")) {
            (Some(value), _) => Some(("const", vec![value.clone()])),
            (None, Some(Value::Array(values))) => Some(("enum", values.clone())),
            (None, Some(_)) => return Err(GrammarError::invalid("enum must be an array")),
            (None, None) => None,
        };
        if let Some((keyword, values)) = literals {
            let values = self.narrowed(without(fields, &[keyword]), values)?;
            if values.is_empty() {
                return Err(GrammarError::invalid(format!(
                    "{keyword} has no value satisfying its sibling assertions"
                )));
            }
            return Ok(Expr::alt(values.iter().map(literal_value)));
        }
        // Multiple oneOf branches need an exclusivity proof, which this
        // compiler does not provide.
        for keyword in ["anyOf", "oneOf"] {
            if let Some(branches) = fields.get(keyword) {
                let Value::Array(branches) = branches else {
                    return Err(GrammarError::invalid(format!("{keyword} must be an array")));
                };
                if keyword == "oneOf" && branches.len() != 1 {
                    return Err(GrammarError::invalid(
                        "oneOf with multiple branches is not supported; use anyOf",
                    ));
                }
                let branches = branches
                    .iter()
                    .map(|branch| self.schema(branch))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Expr::alt(branches));
            }
        }
        if let Some(parts) = fields.get("allOf") {
            return match parts {
                Value::Array(parts) if parts.len() == 1 => self.schema(&parts[0]),
                // General intersections require a different grammar construction.
                Value::Array(_) => Err(GrammarError::invalid("allOf requires exactly one branch")),
                _ => Err(GrammarError::invalid("allOf must be an array")),
            };
        }
        let types: Vec<&str> = match fields.get("type") {
            None => infer_types(fields),
            Some(Value::String(kind)) => vec![kind.as_str()],
            Some(Value::Array(kinds)) => kinds
                .iter()
                .map(|kind| {
                    kind.as_str()
                        .ok_or_else(|| GrammarError::invalid("type entries must be strings"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(GrammarError::invalid("type must be a string or an array")),
        };
        if types.is_empty() {
            return Ok(self.any_value());
        }
        let mut branches = Vec::with_capacity(types.len());
        for kind in types {
            branches.push(match kind {
                "string" => self.string(fields)?,
                "integer" => self.integer(fields)?,
                "number" => self.number(fields)?,
                "boolean" => Expr::alt([Expr::literal("true"), Expr::literal("false")]),
                "null" => Expr::literal("null"),
                "array" => self.array(fields)?,
                "object" => self.object(fields)?,
                other => {
                    return Err(GrammarError::invalid(format!(
                        "unknown JSON schema type '{other}'"
                    )));
                }
            });
        }
        Ok(Expr::alt(branches))
    }

    /// Narrow literals by their sibling assertions, propagating unsupported schemas.
    fn narrowed(
        &self,
        rest: Map<String, Value>,
        values: Vec<Value>,
    ) -> Result<Vec<Value>, GrammarError> {
        if constraints(&rest).next().is_none() {
            return Ok(values);
        }
        let rest = Value::Object(rest);
        // Validate siblings once for the whole enum. Containers follow their
        // keys and values below; scalars share one compiled automaton.
        let grammar = self.literal_grammar(&rest)?;
        let automaton = if values
            .iter()
            .any(|value| !value.is_object() && !value.is_array())
        {
            Some(Automaton::compile_with_budget(&grammar, &self.budget)?)
        } else {
            None
        };
        self.budget.charge(
            values.len(),
            values.len().saturating_mul(std::mem::size_of::<Value>()),
        )?;
        let mut admitted = Vec::new();
        for value in values {
            let accepts = if value.is_object() || value.is_array() {
                self.admits_container(&rest, &value)?
            } else {
                self.admits_scalar(automaton.as_ref().expect("scalar automaton"), &value)?
            };
            if accepts {
                admitted.push(value);
            }
        }
        Ok(admitted)
    }

    fn literal_grammar(&self, schema: &Value) -> Result<Grammar, GrammarError> {
        self.budget.literal_compilation()?;
        let mut builder = GrammarBuilder::new();
        let mut compiler = SchemaCompiler::new(&mut builder, self.document, JsonOptions::default());
        compiler.budget = self.budget.nested()?;
        let expr = compiler.schema(schema)?;
        let root = builder.rule(expr);
        builder.finish(root)
    }

    fn admits_scalar(&self, automaton: &Automaton, value: &Value) -> Result<bool, GrammarError> {
        // Reserve serialization before allocating the text, then charge actual
        // matcher work. Exhaustion is never an ordinary literal rejection.
        self.budget.schema(value)?;
        let text = serde_json::to_string(value).expect("JSON values serialize");
        self.budget.charge(text.len(), text.len())?;
        let mut config = Config::initial_budgeted(automaton, &self.budget);
        for &byte in text.as_bytes() {
            if config.failure.is_some() || config.is_dead() {
                break;
            }
            config = config.step_budgeted(automaton, byte, &self.budget);
        }
        if let Some(error) = config.failure {
            return Err(error);
        }
        Ok(config.complete)
    }

    /// Validate literals as JSON values, not as one ordering of generated keys.
    fn admits(&self, schema: &Value, value: &Value) -> Result<bool, GrammarError> {
        let _depth = self.budget.enter_schema()?;
        self.budget.charge(1, 0)?;
        if let Value::Bool(allowed) = schema {
            return Ok(*allowed);
        }
        let grammar = self.literal_grammar(schema)?;
        if !value.is_object() && !value.is_array() {
            let automaton = Automaton::compile_with_budget(&grammar, &self.budget)?;
            return self.admits_scalar(&automaton, value);
        }
        self.admits_container(schema, value)
    }

    fn admits_container(&self, schema: &Value, value: &Value) -> Result<bool, GrammarError> {
        // Account for normalization, container visits and comparisons before
        // cloning schema fields or traversing literal members.
        self.budget.schema(schema)?;
        self.budget.schema(value)?;
        let applicable = applicable_fields(schema.as_object().expect("compiled object schema"));
        let fields = &applicable;
        if let Some(reference) = fields.get("$ref").and_then(Value::as_str) {
            let target = self
                .document
                .pointer(&reference[1..])
                .expect("compiled local reference");
            return self.admits(target, value);
        }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let Some(branches) = fields.get(keyword).and_then(Value::as_array) {
                let mut matches = 0;
                for branch in branches {
                    matches += usize::from(self.admits(branch, value)?);
                }
                return Ok(match keyword {
                    "anyOf" => matches > 0,
                    "oneOf" => matches == 1,
                    _ => matches == branches.len(),
                });
            }
        }
        if let Some(literal) = fields.get("const") {
            self.budget.schema(value)?;
            self.budget.schema(literal)?;
            if literal != value {
                return Ok(false);
            }
        }
        if let Some(literals) = fields.get("enum").and_then(Value::as_array) {
            let mut found = false;
            for literal in literals {
                self.budget.schema(value)?;
                self.budget.schema(literal)?;
                if literal == value {
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(false);
            }
        }
        let kind = if value.is_object() { "object" } else { "array" };
        if fields.get("type").is_some_and(|types| match types {
            Value::String(t) => t != kind,
            Value::Array(types) => !types.iter().any(|t| t.as_str() == Some(kind)),
            _ => true,
        }) {
            return Ok(false);
        }
        if let Some(object) = value.as_object() {
            if fields
                .get("required")
                .and_then(Value::as_array)
                .is_some_and(|keys| {
                    keys.iter().any(|key| {
                        !object.contains_key(key.as_str().expect("compiled required key"))
                    })
                })
            {
                return Ok(false);
            }
            let properties = fields.get("properties").and_then(Value::as_object);
            for (key, item) in object {
                if let Some(child) = properties
                    .and_then(|p| p.get(key))
                    .or_else(|| fields.get("additionalProperties"))
                    && !self.admits(child, item)?
                {
                    return Ok(false);
                }
            }
        } else if let Some(array) = value.as_array() {
            let min = bound(fields, "minItems", MAX_ITEMS_BOUND)?.unwrap_or(0);
            let max = bound(fields, "maxItems", MAX_ITEMS_BOUND)?;
            if (array.len() as u64) < min || max.is_some_and(|max| array.len() as u64 > max) {
                return Ok(false);
            }
            let legacy = fields.get("items").is_some_and(Value::is_array);
            let prefix = fields
                .get(if legacy { "items" } else { "prefixItems" })
                .and_then(Value::as_array);
            for (index, item) in array.iter().enumerate() {
                if let Some(child) = prefix
                    .and_then(|p| p.get(index))
                    .or_else(|| fields.get(if legacy { "additionalItems" } else { "items" }))
                    && !self.admits(child, item)?
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn integer(&mut self, fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
        let (lower, upper) = numeric_bounds(fields, true)?;
        let divisor = fields
            .get("multipleOf")
            .and_then(|value| {
                value.as_u64().or_else(|| {
                    value
                        .as_f64()
                        .filter(|v| v.fract() == 0.0 && *v > 0.0 && *v < 1e18)
                        .map(|v| v as u64)
                })
            })
            .filter(|d| *d > 0);
        if let Some(divisor) = divisor {
            if let (Some(lower), Some(upper)) = (lower, upper) {
                let low = lower.value + i128::from(!lower.inclusive);
                let high = upper.value - i128::from(!upper.inclusive);
                let divisor = i128::from(divisor);
                let first = low + (-low).rem_euclid(divisor);
                if high < first {
                    return Ok(Expr::Nothing);
                }
                let count = (high - first) / divisor + 1;
                if count <= 4096 {
                    return Ok(Expr::alt(
                        (0..count).map(|i| Expr::literal(&(first + i * divisor).to_string())),
                    ));
                }
            } else if lower.is_none() && upper.is_none() && divisor <= 32 {
                // Decimal remainder automaton. Each transition consumes a digit,
                // so recursive rules cannot form an empty cycle.
                let rules: Vec<_> = (0..divisor).map(|_| self.builder.declare()).collect();
                for remainder in 0..divisor {
                    let mut branches = Vec::new();
                    if remainder == 0 {
                        branches.push(Expr::Empty);
                    }
                    for digit in 0..10 {
                        branches.push(Expr::seq([
                            Expr::literal(&digit.to_string()),
                            Expr::Rule(rules[((remainder * 10 + digit) % divisor) as usize]),
                        ]));
                    }
                    self.builder
                        .define(rules[remainder as usize], Expr::alt(branches));
                }
                let positive = Expr::alt((1..10).map(|digit| {
                    Expr::seq([
                        Expr::literal(&digit.to_string()),
                        Expr::Rule(rules[(digit % divisor) as usize]),
                    ])
                }));
                return Ok(Expr::seq([
                    Expr::literal("-").optional(),
                    Expr::alt([Expr::literal("0"), positive]),
                ]));
            }
        }
        if fields.contains_key("multipleOf") {
            return Err(GrammarError::invalid(
                "multipleOf cannot be enforced with these bounds or divisor",
            ));
        }
        integer(fields)
    }

    fn reference(&mut self, reference: &str) -> Result<Expr, GrammarError> {
        if let Some(&rule) = self.references.get(reference) {
            return Ok(Expr::Rule(rule));
        }
        if reference != "#" && !reference.starts_with("#/") {
            return Err(GrammarError::invalid("$ref must be a local JSON pointer"));
        }
        let pointer = &reference[1..];
        let target = self.document.pointer(pointer).ok_or_else(|| {
            GrammarError::invalid(format!(
                "JSON schema reference '{reference}' does not resolve"
            ))
        })?;
        let rule = self.builder.declare();
        self.references.insert(reference.to_owned(), rule);
        let expr = self.schema(target)?;
        self.builder.define(rule, expr);
        Ok(Expr::Rule(rule))
    }

    fn string(&mut self, fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
        let min = bound(fields, "minLength", MAX_LENGTH_BOUND)?;
        let max = bound(fields, "maxLength", MAX_LENGTH_BOUND)?;
        let pattern = fields.get("pattern");
        let format = fields.get("format");
        if pattern.is_some() || format.is_some() {
            for keyword in ["minLength", "maxLength"] {
                if fields.contains_key(keyword) {
                    return Err(GrammarError::invalid(format!(
                        "{keyword} with pattern or format is not supported"
                    )));
                }
            }
        }
        if pattern.is_some() && format.is_some() {
            return Err(GrammarError::invalid(
                "format with pattern is not supported",
            ));
        }
        let content = if let Some(pattern) = pattern {
            let Value::String(pattern) = pattern else {
                return Err(GrammarError::invalid("pattern must be a string"));
            };
            regex_expr(pattern, &json_characters).map_err(|error| match error {
                error @ GrammarError::TooLarge(_) => error,
                error => GrammarError::invalid(format!("pattern: {error}")),
            })?
        } else if let Some(format) = format {
            let Value::String(format) = format else {
                return Err(GrammarError::invalid("format must be a string"));
            };
            match format_pattern(format) {
                Some(pattern) => regex_expr(&pattern, &json_characters)?,
                None => {
                    return Err(GrammarError::invalid(format!(
                        "unsupported format: {format}"
                    )));
                }
            }
        } else {
            let min = min.unwrap_or(0) as u32;
            let max = max.map(|max| max as u32);
            if max.is_some_and(|max| max < min) {
                return Err(GrammarError::invalid("maxLength is below minLength"));
            }
            string_character().repeat(min, max)
        };
        Ok(Expr::seq([
            Expr::literal("\""),
            content,
            Expr::literal("\""),
        ]))
    }

    fn array(&mut self, fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
        let min = bound(fields, "minItems", MAX_ITEMS_BOUND)?.unwrap_or(0);
        let max = bound(fields, "maxItems", MAX_ITEMS_BOUND)?;
        let legacy_tuple = fields.get("items").is_some_and(Value::is_array);
        let prefix = match fields
            .get("prefixItems")
            .or_else(|| legacy_tuple.then(|| &fields["items"]))
        {
            None => Vec::new(),
            Some(Value::Array(schemas)) => schemas
                .iter()
                .map(|schema| self.schema(schema))
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(GrammarError::invalid("prefixItems must be an array")),
        };
        let items = match fields.get(if legacy_tuple {
            "additionalItems"
        } else {
            "items"
        }) {
            None => Some(self.any_value()),
            Some(Value::Bool(false)) => None,
            Some(schema) => Some(self.schema(schema)?),
        };
        if items.is_none() && min > prefix.len() as u64 {
            return Err(GrammarError::invalid(
                "minItems exceeds the number of permitted items",
            ));
        }
        for item in &prefix {
            self.budget.expr_copies(item, 2)?;
        }
        if let Some(item) = &items {
            // One persistent copy and the temporary alternation copies used
            // for optional empty arrays. Reserve before any clone occurs.
            self.budget.expr_copies(item, 3)?;
        }
        Ok(self.array_of(&prefix, items, min, max))
    }

    /// Every prefix item, then `items` until the counts are met.
    fn array_of(&self, prefix: &[Expr], items: Option<Expr>, min: u64, max: Option<u64>) -> Expr {
        if max.is_some_and(|max| max < min || max < prefix.len() as u64) {
            return Expr::Nothing;
        }
        let prefix: Vec<_> = prefix
            .iter()
            .map(|item| Expr::seq([item.clone(), self.whitespace()]))
            .collect();
        let items = items.map(|item| Expr::seq([item, self.whitespace()]));
        let rest_min = min.saturating_sub(prefix.len() as u64) as u32;
        let rest_max = max.map(|max| (max - prefix.len() as u64) as u32);
        let comma = self.comma();
        let body = if prefix.is_empty() {
            match items {
                None => Expr::Empty,
                Some(_) if rest_max == Some(0) => Expr::Empty,
                Some(item) if rest_min == 0 => Expr::seq([
                    item.clone(),
                    Expr::seq([comma, item]).repeat(0, rest_max.map(|max| max - 1)),
                ])
                .optional(),
                Some(item) => Expr::seq([
                    item.clone(),
                    Expr::seq([comma, item]).repeat(rest_min - 1, rest_max.map(|max| max - 1)),
                ]),
            }
        } else {
            let mut parts = Vec::new();
            for (index, item) in prefix.iter().enumerate() {
                if index > 0 {
                    parts.push(comma.clone());
                }
                parts.push(item.clone());
            }
            match items {
                Some(item) => {
                    parts.push(Expr::seq([comma, item]).repeat(rest_min, rest_max));
                }
                None if rest_min > 0 => return Expr::Nothing,
                None => {}
            }
            Expr::seq(parts)
        };
        Expr::seq([
            Expr::literal("["),
            self.whitespace(),
            body,
            Expr::literal("]"),
        ])
    }

    fn object(&mut self, fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
        let required: Vec<&str> = match fields.get("required") {
            None => Vec::new(),
            Some(Value::Array(names)) => names
                .iter()
                .map(|name| {
                    name.as_str()
                        .ok_or_else(|| GrammarError::invalid("required entries must be strings"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(GrammarError::invalid("required must be an array")),
        };
        let properties = match fields.get("properties") {
            None => None,
            Some(Value::Object(properties)) => Some(properties),
            Some(_) => return Err(GrammarError::invalid("properties must be an object")),
        };
        let mut members = Vec::new();
        for (name, schema) in properties.into_iter().flatten() {
            let value = self.schema(schema)?;
            let member = Expr::seq([
                literal_value(&Value::String(name.clone())),
                self.colon(),
                value,
            ]);
            members.push((member, required.contains(&name.as_str())));
        }
        // Guidance permits required names without a properties entry.
        for name in &required {
            if !properties.is_some_and(|properties| properties.contains_key(*name)) {
                let value = match fields.get("additionalProperties") {
                    _ if fields.contains_key("patternProperties") => self.any_value(),
                    Some(Value::Bool(false)) => return Ok(Expr::Nothing),
                    Some(schema) => self.schema(schema)?,
                    None => self.any_value(),
                };
                members.push((
                    Expr::seq([
                        literal_value(&Value::String((*name).to_owned())),
                        self.colon(),
                        value,
                    ]),
                    true,
                ));
            }
        }
        let additional = match fields.get("additionalProperties") {
            _ if fields.contains_key("patternProperties") => Some(self.any_value()),
            None if properties.is_some() => None,
            None | Some(Value::Bool(true)) => Some(self.any_value()),
            Some(Value::Bool(false)) => None,
            Some(schema) => Some(self.schema(schema)?),
        };
        Ok(self.object_of(members, additional))
    }

    /// `{`, the members in order, additional string-keyed members, `}`.
    fn object_of(&self, members: Vec<(Expr, bool)>, additional: Option<Expr>) -> Expr {
        let members = members
            .into_iter()
            .map(|(member, required)| (Expr::seq([member, self.whitespace()]), required))
            .collect();
        let tail = additional.map(|value| {
            Box::new(Expr::seq([
                string_expr(),
                self.colon(),
                value,
                self.whitespace(),
            ]))
        });
        Expr::seq([
            Expr::literal("{"),
            self.whitespace(),
            Expr::OrderedSubset {
                items: members,
                tail,
                separator: Box::new(self.comma()),
            },
            Expr::literal("}"),
        ])
    }
}

/// Remove only assertions proven irrelevant to every explicit instance type,
/// or assertions whose value imposes no restriction on any instance.
fn applicable_fields(fields: &Map<String, Value>) -> Map<String, Value> {
    let types: Option<Vec<&str>> = match fields.get("type") {
        Some(Value::String(kind)) => Some(vec![kind.as_str()]),
        Some(Value::Array(kinds)) if !kinds.is_empty() => kinds.iter().map(Value::as_str).collect(),
        _ => None,
    };
    let groups: &[(&[&str], &[&str])] = &[
        (
            &["number", "integer"],
            &[
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            ],
        ),
        (
            &["string"],
            &[
                "minLength",
                "maxLength",
                "pattern",
                "format",
                "contentEncoding",
                "contentMediaType",
                "contentSchema",
            ],
        ),
        (
            &["array"],
            &[
                "items",
                "prefixItems",
                "additionalItems",
                "minItems",
                "maxItems",
                "uniqueItems",
                "contains",
                "minContains",
                "maxContains",
                "unevaluatedItems",
            ],
        ),
        (
            &["object"],
            &[
                "properties",
                "additionalProperties",
                "required",
                "minProperties",
                "maxProperties",
                "patternProperties",
                "propertyNames",
                "dependencies",
                "dependentRequired",
                "dependentSchemas",
                "unevaluatedProperties",
            ],
        ),
    ];
    let unconstrained = |schema: &Value| {
        schema == &Value::Bool(true)
            || schema
                .as_object()
                .is_some_and(|fields| constraints(fields).next().is_none())
    };
    fields
        .iter()
        .filter(|(key, value)| {
            let vacuous = match key.as_str() {
                "uniqueItems" | "not" => **value == Value::Bool(false),
                "minLength" | "minItems" | "minProperties" => value.as_f64() == Some(0.0),
                "required" => value.as_array().is_some_and(Vec::is_empty),
                "allOf" => value
                    .as_array()
                    .is_some_and(|branches| branches.iter().all(&unconstrained)),
                "anyOf" => value
                    .as_array()
                    .is_some_and(|branches| branches.iter().any(&unconstrained)),
                "propertyNames" | "unevaluatedProperties" | "unevaluatedItems" => {
                    unconstrained(value)
                }
                "dependentRequired" | "dependentSchemas" | "dependencies" | "patternProperties" => {
                    value.as_object().is_some_and(Map::is_empty)
                }
                "contains" => {
                    fields.get("minContains").and_then(Value::as_f64) == Some(0.0)
                        && !fields.contains_key("maxContains")
                }
                "minContains" => {
                    !fields.contains_key("contains")
                        || (value.as_f64() == Some(0.0) && !fields.contains_key("maxContains"))
                }
                "maxContains" => !fields.contains_key("contains"),
                "pattern" => value.as_str() == Some(""),
                _ => false,
            };
            !vacuous
                && !types.as_ref().is_some_and(|types| {
                    groups.iter().any(|(kinds, keys)| {
                        keys.contains(&key.as_str())
                            && types.iter().all(|kind| !kinds.contains(kind))
                    })
                })
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Keywords that constrain a value, as opposed to annotations.
fn constraints(fields: &Map<String, Value>) -> impl Iterator<Item = &str> {
    fields
        .keys()
        .map(String::as_str)
        .filter(|key| !ANNOTATIONS.contains(key))
}

fn without(fields: &Map<String, Value>, keys: &[&str]) -> Map<String, Value> {
    fields
        .iter()
        .filter(|(key, _)| !keys.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn infer_types(fields: &serde_json::Map<String, Value>) -> Vec<&'static str> {
    let has = |keys: &[&str]| keys.iter().any(|key| fields.contains_key(*key));
    let mut types = Vec::new();
    if has(&["properties", "additionalProperties", "required"]) {
        types.push("object");
    }
    if has(&["items", "prefixItems", "minItems", "maxItems"]) {
        types.push("array");
    }
    if has(&["minLength", "maxLength", "pattern", "format"]) {
        types.push("string");
    }
    if has(&["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]) {
        types.push("number");
    }
    types
}

fn bound(
    fields: &serde_json::Map<String, Value>,
    key: &str,
    limit: u64,
) -> Result<Option<u64>, GrammarError> {
    match fields.get(key) {
        None => Ok(None),
        Some(Value::Number(number)) => {
            let value = number
                .as_u64()
                .or_else(|| {
                    number
                        .as_f64()
                        .filter(|v| v.fract() == 0.0 && *v >= 0.0)
                        .map(|v| v as u64)
                })
                .ok_or_else(|| {
                    GrammarError::invalid(format!("{key} must be a non-negative integer"))
                })?;
            if value > limit {
                return Err(GrammarError::invalid(format!(
                    "{key} exceeds supported limit {limit}"
                )));
            }
            Ok(Some(value))
        }
        Some(_) => Err(GrammarError::invalid(format!("{key} must be a number"))),
    }
}

/// Numeric bounds, rounded inward for integers.
fn numeric_bounds(
    fields: &serde_json::Map<String, Value>,
    integral: bool,
) -> Result<(Option<Bound>, Option<Bound>), GrammarError> {
    let read = |key: &str| -> Result<Option<f64>, GrammarError> {
        match fields.get(key) {
            None => Ok(None),
            Some(Value::Number(number)) => number
                .as_f64()
                .filter(|value| value.is_finite() && value.abs() < 9_007_199_254_740_992.0)
                .filter(|value| integral || value.fract() == 0.0)
                .map(Some)
                .ok_or_else(|| {
                    GrammarError::invalid(format!("{key} is outside supported numeric bounds"))
                }),
            Some(_) => Err(GrammarError::invalid(format!("{key} must be a number"))),
        }
    };
    let mut lower: Option<(f64, bool)> = read("minimum")?.map(|value| (value, true));
    if let Some(value) = read("exclusiveMinimum")?
        && lower.is_none_or(|(current, _)| value >= current)
    {
        lower = Some((value, false));
    }
    let mut upper: Option<(f64, bool)> = read("maximum")?.map(|value| (value, true));
    if let Some(value) = read("exclusiveMaximum")?
        && upper.is_none_or(|(current, _)| value <= current)
    {
        upper = Some((value, false));
    }
    let convert = |bound: Option<(f64, bool)>, ceil: bool| -> Result<Option<Bound>, GrammarError> {
        let Some((value, inclusive)) = bound else {
            return Ok(None);
        };
        if value.fract() == 0.0 {
            return Ok(Some(Bound {
                value: value as i128,
                inclusive,
            }));
        }
        if !integral {
            return Ok(Some(Bound {
                value: if ceil { value.floor() } else { value.ceil() } as i128,
                inclusive: true,
            }));
        }
        let rounded = if ceil { value.ceil() } else { value.floor() };
        Ok(Some(Bound {
            value: rounded as i128,
            inclusive: true,
        }))
    };
    Ok((convert(lower, true)?, convert(upper, false)?))
}

fn integer(fields: &serde_json::Map<String, Value>) -> Result<Expr, GrammarError> {
    let (lower, upper) = numeric_bounds(fields, true)?;
    if lower.is_none() && upper.is_none() {
        return Ok(integer_expr());
    }
    Ok(integer_range(lower, upper))
}

fn integer_expr() -> Expr {
    Expr::seq([
        Expr::literal("-").optional(),
        Expr::alt([
            Expr::literal("0"),
            Expr::seq([
                Expr::Class(CharClass::from_ranges([('1' as u32, '9' as u32)])),
                Expr::Class(digits()).star(),
            ]),
        ]),
    ])
}

fn number_expr() -> Expr {
    Expr::seq([
        integer_expr(),
        Expr::seq([Expr::literal("."), Expr::Class(digits()).plus()]).optional(),
        Expr::seq([
            Expr::Class(CharClass::of("eE")),
            Expr::Class(CharClass::of("+-")).optional(),
            Expr::Class(digits()).plus(),
        ])
        .optional(),
    ])
}

/// The most integer digits a finite number admits with a fraction or a
/// negative exponent ([`SchemaCompiler::finite_numbers`]), so its magnitude
/// stays below 1e308.
const FINITE_INTEGER_DIGITS: u32 = 308;

/// The most integer digits a finite number admits with a positive exponent:
/// the significant digits a binary64 holds.
const FINITE_SCALED_DIGITS: u32 = 17;

/// [`number_expr`] less the numbers a binary64 cannot hold. An integer keeps
/// every digit. A number with a fraction or exponent is admitted when it is
/// zero or its magnitude is below 1e308, under any negative exponent since
/// underflow is finite. The boundary is conservative by less than one decade:
/// magnitudes from 1e308 to the largest finite 1.7976931348623157e308 are
/// refused with those that overflow.
///
/// Three rarer finite spellings are refused too. A finite automaton cannot
/// weigh an unbounded digit count against an unbounded exponent, which rules
/// out an integer part of more than 308 digits with a negative exponent and a
/// fraction below 1 with a positive exponent above 308, such as `0.001e309`.
/// And an integer part under a positive exponent has at most
/// [`FINITE_SCALED_DIGITS`] digits: weighing each of 308 lengths against the
/// exponent would multiply the automaton of every number.
fn finite_number_expr() -> Expr {
    let digit = || Expr::Class(digits());
    let nonzero = || Expr::Class(CharClass::from_ranges([('1' as u32, '9' as u32)]));
    let fraction = || Expr::seq([Expr::literal("."), digit().plus()]);
    let mark = || Expr::Class(CharClass::of("eE"));
    let negative = || Expr::seq([mark(), Expr::literal("-"), digit().plus()]);
    // A positive exponent of at most `high`, with any leading zeros.
    let positive = |high: u32| {
        Expr::seq([
            mark(),
            Expr::literal("+").optional(),
            Expr::literal("0").star(),
            unsigned_range(0, Some(u128::from(high))),
        ])
    };
    // After `length` integer digits, a positive exponent reaches at most
    // 308 - `length`, which keeps the magnitude below 1e308.
    let mut scaled = Expr::Nothing;
    for length in (1..=FINITE_SCALED_DIGITS).rev() {
        let first = if length == 1 { nonzero() } else { digit() };
        let exponent = positive(FINITE_INTEGER_DIGITS - length);
        scaled = Expr::seq([
            first,
            Expr::alt([Expr::seq([fraction().optional(), exponent]), scaled]),
        ]);
    }
    let integer_part = Expr::alt([
        Expr::literal("0"),
        Expr::seq([
            nonzero(),
            digit().repeat(0, Some(FINITE_INTEGER_DIGITS - 1)),
        ]),
    ]);
    Expr::alt([
        integer_expr(),
        Expr::seq([
            Expr::literal("-").optional(),
            Expr::alt([
                Expr::seq([
                    integer_part,
                    Expr::alt([Expr::seq([fraction(), negative().optional()]), negative()]),
                ]),
                scaled,
                // A fraction below 1 under a positive exponent of at most 308.
                Expr::seq([
                    Expr::literal("0"),
                    fraction().optional(),
                    positive(FINITE_INTEGER_DIGITS),
                ]),
                // Zero under any exponent.
                Expr::seq([
                    Expr::literal("0"),
                    Expr::seq([Expr::literal("."), Expr::literal("0").plus()]).optional(),
                    mark(),
                    Expr::Class(CharClass::of("+-")).optional(),
                    digit().plus(),
                ]),
            ]),
        ]),
    ])
}

/// Characters that appear in a JSON string without an escape.
fn unescaped() -> CharClass {
    CharClass::of("\"\\")
        .union(&CharClass::from_ranges([(0, 0x1F)]))
        .complement()
}

/// One JSON string character: unescaped, or any JSON escape.
fn string_character() -> Expr {
    let hex = Expr::Class(CharClass::from_ranges([
        ('0' as u32, '9' as u32),
        ('a' as u32, 'f' as u32),
        ('A' as u32, 'F' as u32),
    ]));
    Expr::alt([
        Expr::Class(unescaped()),
        Expr::seq([
            Expr::literal("\\"),
            Expr::alt([
                Expr::Class(CharClass::of("\"\\/bfnrt")),
                Expr::seq([Expr::literal("u"), hex.repeat(4, Some(4))]),
            ]),
        ]),
    ])
}

fn string_expr() -> Expr {
    Expr::seq([
        Expr::literal("\""),
        string_character().star(),
        Expr::literal("\""),
    ])
}

/// A pattern class inside a JSON string: characters that need an escape are
/// written with their canonical escape.
fn json_characters(class: CharClass) -> Expr {
    let plain = class.intersection(&unescaped());
    let mut branches = Vec::new();
    if !plain.is_empty() {
        branches.push(Expr::Class(plain));
    }
    for &(lo, hi) in class.intersection(&unescaped().complement()).ranges() {
        for scalar in lo..=hi {
            let character = char::from_u32(scalar).expect("scalar value");
            let encoded = serde_json::to_string(&character.to_string()).expect("string encodes");
            branches.push(Expr::literal(&encoded[1..encoded.len() - 1]));
        }
    }
    Expr::alt(branches)
}

/// The compact JSON text of a value.
fn literal_value(value: &Value) -> Expr {
    Expr::literal(&serde_json::to_string(value).expect("JSON values serialize"))
}

fn format_pattern(format: &str) -> Option<String> {
    const DATE: &str = concat!(
        r"(([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3})-",
        r"((01|03|05|07|08|10|12)-(0[1-9]|[12][0-9]|3[01])|",
        r"(04|06|09|11)-(0[1-9]|[12][0-9]|30)|02-(0[1-9]|1[0-9]|2[0-8]))|",
        r"([0-9]{2}(0[48]|[2468][048]|[13579][26])|(0[48]|[2468][048]|[13579][26])00)-02-29)"
    );
    const TIME: &str = r"([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])";
    Some(match format {
        "date" => DATE,
        "time" => TIME,
        "date-time" => return Some(format!("{DATE}[Tt]{TIME}")),
        // A permissive email check includes quoted and international addresses.
        "email" | "idn-email" => r"[\s\S]*@[\s\S]*",
        "uuid" => r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "ipv4" => {
            r"((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
        }
        _ => return None,
    }.to_owned())
}
