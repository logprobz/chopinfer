use runtime_grammar::{JsonOptions, json_schema_grammar};
use serde_json::json;

#[test]
fn unenforced_assertions_are_refused_with_the_keyword() {
    for (keyword, schema) in [
        (
            "minItems",
            json!({"type":"array", "items":false, "minItems":1}),
        ),
        ("minimum", json!({"type":"number","minimum":0.5})),
        ("maximum", json!({"type":"number","maximum":1e20})),
        ("minLength", json!({"type":"string","minLength":5000})),
        ("minItems", json!({"type":"array","minItems":2000})),
        (
            "multipleOf",
            json!({"type":"integer","minimum":0,"multipleOf":7}),
        ),
        ("multipleOf", json!({"type":"integer","multipleOf":33})),
        (
            "multipleOf",
            json!({"type":"integer","minimum":0,"maximum":10000,"multipleOf":2}),
        ),
        ("multipleOf", json!({"type":"number","multipleOf":0.1})),
        (
            "maxLength",
            json!({"type":"string","pattern":"^a+$","maxLength":2}),
        ),
        (
            "minLength",
            json!({"type":"string","format":"date","minLength":12}),
        ),
        ("format", json!({"type":"string","format":"custom"})),
        ("pattern", json!({"type":"string","pattern":"^(?!bad).*$"})),
        ("minProperties", json!({"type":"object","minProperties":3})),
        ("not", json!({"not":{"type":"string"}})),
        ("uniqueItems", json!({"type":"array","uniqueItems":true})),
        ("if", json!({"if":{"type":"string"}})),
        ("$ref", json!({"$ref":"https://example.com/schema"})),
        (
            "minimum",
            json!({"$ref":"#/$defs/n","minimum":2,"$defs":{"n":{"type":"integer"}}}),
        ),
        ("allOf", json!({"allOf":[{"type":"integer"},{"minimum":2}]})),
        (
            "oneOf",
            json!({"oneOf":[{"type":"integer"},{"type":"number"}]}),
        ),
        ("not", json!({"const":1,"not":{"const":1}})),
        (
            "not",
            json!({"type":"object","properties":{"x":{"not":{}}}}),
        ),
    ] {
        let result = json_schema_grammar(&schema, JsonOptions::default());
        assert!(result.is_err(), "silently accepted {schema}");
        assert!(
            result.unwrap_err().to_string().contains(keyword),
            "missing {keyword}: {schema}"
        );
    }
}
