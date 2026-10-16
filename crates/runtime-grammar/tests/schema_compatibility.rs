mod common;

use std::io::Write;
use std::process::{Command, Stdio};

use common::{Rng, adversarial_vocabulary, compile, random_walk};
use runtime_grammar::{GrammarMatcher, JsonOptions, json_schema_grammar};
use serde_json::{Value, json};

fn matches(schema: &Value, text: &str) -> bool {
    let grammar = json_schema_grammar(schema, JsonOptions::default()).unwrap();
    let mut matcher = GrammarMatcher::new(compile(&grammar, &adversarial_vocabulary()));
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

#[test]
fn supported_keywords_and_formats_compile() {
    for schema in [
        json!({"type": "integer", "multipleOf": 2}),
        json!({"type": "array", "items": [{"type": "integer"}], "additionalItems": false}),
        json!({"type": "object", "required": ["name"]}),
    ] {
        let grammar = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
        compile(&grammar, &adversarial_vocabulary());
    }
    for format in [
        "email",
        "idn-email",
        "date",
        "time",
        "date-time",
        "ipv4",
        "uuid",
    ] {
        json_schema_grammar(
            &json!({"type":"string", "format":format}),
            JsonOptions::default(),
        )
        .unwrap();
    }
}

#[test]
fn literal_narrowing_never_restores_rejected_values() {
    let schema = json!({"type":"integer", "enum":[1,2,4], "multipleOf":2});
    assert!(!matches(&schema, "1"));
    assert!(matches(&schema, "2"));
    assert!(matches(&schema, "4"));
    let schema = json!({"type":"integer", "enum":[1,3], "multipleOf":2});
    assert!(json_schema_grammar(&schema, JsonOptions::default()).is_err());
    assert!(
        json_schema_grammar(
            &json!({"const":1,"minimum":"invalid"}),
            JsonOptions::default()
        )
        .is_err()
    );
}

#[test]
fn integer_multiples_enforce_divisibility() {
    for divisor in [1, 2, 3, 7, 10, 32] {
        let schema = json!({"type": "integer", "multipleOf": divisor});
        let grammar = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
        let compiled = compile(&grammar, &adversarial_vocabulary());
        for value in -100..=100 {
            let mut matcher = GrammarMatcher::new(compiled.clone());
            let accepted =
                matcher.accept_bytes(value.to_string().as_bytes()).is_ok() && matcher.is_complete();
            assert_eq!(accepted, value % divisor == 0, "{schema}: {value}");
        }
    }
    let decimal_divisor = json!({"type": "integer", "multipleOf": 2.0});
    assert!(matches(&decimal_divisor, "4"));
    assert!(!matches(&decimal_divisor, "3"));
    let schema = json!({"type": "integer", "multipleOf": 3, "exclusiveMinimum": -7, "maximum": 7});
    for value in -10..=10 {
        assert_eq!(
            matches(&schema, &value.to_string()),
            (-6..=6).contains(&value) && value % 3 == 0
        );
    }
}

#[test]
fn generated_values_pass_python_jsonschema() {
    let schemas = [
        json!({"type": "integer", "multipleOf": 7, "minimum": -100, "maximum": 100}),
        json!({"type": "integer", "multipleOf": 3}),
        json!({"type": "string", "format": "date"}),
        json!({"type": "string", "format": "uuid"}),
        json!({"type": "string", "format": "ipv4"}),
        json!({"type": "string", "format": "email"}),
    ];
    let vocabulary = adversarial_vocabulary();
    let mut rows = Vec::new();
    for (index, schema) in schemas.iter().enumerate() {
        let grammar = json_schema_grammar(schema, JsonOptions::default()).unwrap();
        let compiled = compile(&grammar, &vocabulary);
        let mut rng = Rng::new(index as u64 + 10);
        let mut count = 0;
        for _ in 0..200 {
            let (text, complete) = random_walk(&compiled, &mut rng, 150, false);
            if complete {
                rows.push(json!({"schema": schema, "text": String::from_utf8(text).unwrap()}));
                count += 1;
                if count == 20 {
                    break;
                }
            }
        }
        assert_eq!(count, 20, "no completed samples for {schema}");
    }
    // A real validator, independent of the compiler and our test helpers.
    // A missing dependency is a failure, never a silent skip.
    let mut child = Command::new("python3")
        .args([
            "-B",
            "-c",
            r#"
import json, sys
from jsonschema import Draft202012Validator, FormatChecker
rows = json.load(sys.stdin)
assert len(rows) == 120
for row in rows:
    validator = Draft202012Validator(row['schema'], format_checker=FormatChecker())
    validator.validate(json.loads(row['text']))
"#,
        ])
        .stdin(Stdio::piped())
        .spawn()
        .expect("python3 with jsonschema is required");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&rows).unwrap())
        .unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn dates_observe_calendar_and_leap_years() {
    let schema = json!({"type": "string", "format": "date"});
    for date in ["2024-02-29", "2000-02-29", "0001-01-01", "9999-12-31"] {
        assert!(matches(&schema, &serde_json::to_string(date).unwrap()));
    }
    for date in ["1900-02-29", "2023-02-29", "2024-04-31", "0000-01-01"] {
        assert!(!matches(&schema, &serde_json::to_string(date).unwrap()));
    }
}

#[test]
fn client_schemas_compile() {
    // Pydantic v2 discriminated unions, nullable fields, defaults, and nested refs.
    let pydantic = json!({"$defs": {
        "Role": {"enum": ["admin", "user"], "title": "Role", "type": "string"},
        "Cat": {"type": "object", "properties": {"kind": {"const": "cat"}, "lives": {"type": "integer"}}, "required": ["kind", "lives"]},
        "Dog": {"type": "object", "properties": {"kind": {"const": "dog"}}, "required": ["kind"]},
        "User": {"type": "object", "properties": {"role": {"$ref": "#/$defs/Role"}, "email": {"type": "string", "format": "email"}}, "required": ["role", "email"]}
    }, "type": "object", "properties": {
        "owner": {"$ref": "#/$defs/User"},
        "pet": {"discriminator": {"propertyName": "kind", "mapping": {"cat": "#/$defs/Cat", "dog": "#/$defs/Dog"}}, "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}]},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
    }, "required": ["owner", "pet"]});
    let weather = json!({"type": "object", "properties": {"location": {"type": "string", "description": "City and state"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"], "additionalProperties": false});
    let query = json!({"type": "object", "properties": {"table_name": {"type": "string", "enum": ["orders"]}, "columns": {"type": "array", "items": {"type": "string", "enum": ["id", "status", "ordered_at"]}}, "conditions": {"type": "array", "items": {"type": "object", "properties": {"column": {"type": "string"}, "operator": {"enum": ["=", ">", "<"]}, "value": {"anyOf": [{"type": "string"}, {"type": "number"}]}}, "required": ["column", "operator", "value"], "additionalProperties": false}}}, "required": ["table_name", "columns", "conditions"], "additionalProperties": false});
    assert!(json_schema_grammar(&pydantic, JsonOptions::default()).is_err());
    for schema in [weather, query] {
        let grammar = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
        compile(&grammar, &adversarial_vocabulary());
    }
}

#[test]
fn object_literals_ignore_key_order_recursively() {
    for literal in [
        json!({"b":2,"a":{"y":2,"x":1}}),
        json!({"a":{"x":1,"y":2},"b":2}),
    ] {
        for keyword in ["const", "enum"] {
            let mut schema = json!({"type":"object","properties":{"a":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"],"additionalProperties":false},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false});
            schema["allOf"] = json!([]);
            schema[keyword] = if keyword == "const" {
                literal.clone()
            } else {
                json!([literal.clone(), {"b":"bad","a":{"x":1,"y":2}}])
            };
            assert!(matches(&schema, &literal.to_string()));
            assert!(!matches(&schema, r#"{"b":"bad","a":{"x":1,"y":2}}"#));
        }
    }
}

#[test]
fn vacuous_and_inapplicable_assertions_are_accepted() {
    for (schema, text) in [
        (
            json!({"type":"object", "propertyNames":true, "dependentRequired":{}, "unevaluatedProperties":true}),
            "{}",
        ),
        (
            json!({"type":"array", "unevaluatedItems":true, "contains":false, "minContains":0}),
            "[1]",
        ),
        (json!({"type":"string", "allOf":[true,{}]}), "\"x\""),
        (json!({"type":["integer","null"],"multipleOf":2}), "2"),
        (json!({"type":["integer","null"],"multipleOf":2}), "null"),
        (json!({"type":"integer","uniqueItems":false}), "1"),
        (
            json!({"type":"string","minimum":0.5,"multipleOf":2}),
            "\"x\"",
        ),
        (
            json!({"type":["string","null"],"uniqueItems":true,"minItems":2000}),
            "null",
        ),
        (json!({"type":"array","uniqueItems":false}), "[]"),
        (json!({"type":"object","minProperties":0}), "{}"),
    ] {
        assert!(matches(&schema, text), "{schema}");
    }
    for schema in [
        json!({"type":["string","array"],"uniqueItems":true}),
        json!({"type":["string","number"],"multipleOf":2}),
        json!({"uniqueItems":true}),
    ] {
        assert!(json_schema_grammar(&schema, JsonOptions::default()).is_err());
    }
}
