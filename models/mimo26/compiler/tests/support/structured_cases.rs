//! The `tools/qual/structured.py` constraints with outputs the model could
//! write, for the mask benchmark and its regression test.

use std::sync::Arc;

#[path = "frontend.rs"]
mod frontend_support;
use frontend_support::frontend;
use runtime_grammar::{
    Grammar, JsonOptions, TokenVocabulary, VocabularyToken, choice_grammar, json_object_grammar,
    json_schema_grammar, regex_grammar,
};
use runtime_server::{FrontendJsonV1, FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1};
use runtime_tokenizer::Tokenizer;
use serde_json::{Value, json};

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
            "note": {"type": "null"}
        }
    })
}

const PERSON_TEXT: &str = "{\n  \"name\": \"Ada Lovelace\",\n  \"age\": 36,\n  \"status\": \"active\",\n  \
     \"kind\": \"person\",\n  \"address\": {\n    \"city\": \"Paris\",\n    \"zip\": \"75001\"\n  },\n  \
     \"tags\": [\"mathematician\", \"writer\", \"pioneer\"],\n  \"active\": true,\n  \
     \"nickname\": \"Enchantress\",\n  \"note\": null\n}";

const OBJECT_TEXT: &str = "{\n  \"name\": \"Paris\",\n  \"country\": \"France\",\n  \"population\": 2102650,\n  \
     \"landmarks\": [\"Eiffel Tower\", \"Louvre Museum\", \"Notre-Dame de Paris\"],\n  \
     \"transport\": {\n    \"airports\": [\"Charles de Gaulle (CDG)\", \"Orly Airport (ORY)\"],\n    \
     \"public_transit\": [\"Paris M\u{e9}tro\", \"RER (R\u{e9}seau Express R\u{e9}gional)\", \"Bus\"]\n  },\n  \
     \"economy\": {\"gdp_nominal_usd_billion\": 936.1, \"capital\": true, \"note\": null}\n}";

fn tool(name: &str, parameters: Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: name.to_owned(),
        description: None,
        parameters: serde_json::from_value::<FrontendJsonV1>(parameters).expect("tool schema"),
        parameters_present: true,
        strict: None,
    }
}

fn tools() -> Vec<FrontendToolV1> {
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "additionalProperties": false, "required": ["city"],
                   "properties": {"city": {"type": "string"},
                                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}),
        ),
        tool(
            "get_time",
            json!({"type": "object", "additionalProperties": false, "required": ["timezone"],
                   "properties": {"timezone": {"type": "string"}}}),
        ),
    ]
}

const TOOL_TEXT: &str = "<tool_call><function=get_weather><parameter=city>Paris</parameter>\
     <parameter=unit>celsius</parameter></function></tool_call><tool_call><function=get_time>\
     <parameter=timezone>Asia/Tokyo</parameter></function></tool_call>";

pub const STOP_TEXT: &str = "<|im_end|>";

pub struct Case {
    pub name: &'static str,
    pub grammar: Grammar,
    pub text: &'static str,
}

pub fn cases() -> Vec<Case> {
    let frontend = frontend();
    let tool_grammar = |choice| {
        frontend
            .tool_constraint(&tools(), &choice)
            .expect("tool grammar")
            .expect("forced choice constrains")
    };
    let schema = person();
    vec![
        Case {
            name: "json_object",
            grammar: json_object_grammar(JsonOptions::default()),
            text: OBJECT_TEXT,
        },
        Case {
            name: "json_schema_strict",
            grammar: json_schema_grammar(&schema, JsonOptions::default()).expect("schema"),
            text: PERSON_TEXT,
        },
        Case {
            name: "json_schema_compact",
            grammar: json_schema_grammar(
                &schema,
                JsonOptions {
                    any_whitespace: false,
                    ..JsonOptions::default()
                },
            )
            .expect("schema"),
            text: "{\"name\": \"Ada\", \"age\": 36, \"status\": \"pending\", \"kind\": \"person\", \
                   \"address\": {\"city\": \"Paris\", \"zip\": \"75001\"}, \"tags\": [\"a\"], \
                   \"active\": false, \"nickname\": null, \"note\": null}",
        },
        Case {
            name: "choice",
            grammar: choice_grammar(&["positive".into(), "negative".into(), "neutral".into()])
                .expect("choice"),
            text: "positive",
        },
        Case {
            name: "regex",
            grammar: regex_grammar(r"\d{3}-\d{4}").expect("regex"),
            text: "555-0123",
        },
        Case {
            name: "tool_required",
            grammar: tool_grammar(FrontendToolChoiceV1::Required),
            text: TOOL_TEXT,
        },
    ]
}

pub fn vocabulary(tokenizer: &Tokenizer, width: usize) -> Arc<TokenVocabulary> {
    let mut tokens = Vec::new();
    while let Ok(bytes) = tokenizer.token_bytes(tokens.len() as u32) {
        tokens.push(VocabularyToken {
            bytes: bytes.to_vec(),
            special: tokenizer
                .token_is_special(tokens.len() as u32)
                .expect("in range"),
        });
    }
    Arc::new(TokenVocabulary::new(width, tokens).expect("vocabulary"))
}

/// Special tokens the case texts spell.
pub const SPECIALS: [&str; 2] = ["<tool_call>", "</tool_call>"];
