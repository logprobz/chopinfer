mod common;

use std::sync::Arc;

use common::{
    Rng, SPECIAL_OPEN, STOP, adversarial_vocabulary, allowed, bit, brute_force, compile,
    parse_json, random_walk, validates,
};
use runtime_grammar::{
    CompiledGrammar, ConstraintState, Expr, Grammar, GrammarBuilder, GrammarError, GrammarMatcher,
    JsonOptions, SchemaCompiler, TokenVocabulary, VocabularyToken, choice_grammar,
    json_object_grammar, json_schema_grammar, regex_grammar,
};
use serde_json::{Value, json};

fn matches(grammar: &Grammar, text: &str) -> bool {
    let vocabulary = adversarial_vocabulary();
    let mut matcher = GrammarMatcher::new(compile(grammar, &vocabulary));
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

fn schema_grammar(schema: &Value) -> Grammar {
    json_schema_grammar(schema, JsonOptions::default()).unwrap()
}

#[test]
fn json_object_admits_objects_only() {
    let grammar = json_object_grammar(JsonOptions::default());
    for good in [
        "{}",
        "{ }",
        r#"{"a":1}"#,
        "{\n  \"a\" : [1, 2.5e-3, {\"b\": null}],\t\"c\": \"x\\u00e9\\n\"}",
        r#"{"é":"😀"}"#,
        r#"{"a":{"b":{"c":{"d":[[[]]]}}}}"#,
    ] {
        assert!(matches(&grammar, good), "{good}");
    }
    for bad in [
        "[]",
        " {}",
        "{} ",
        r#"{"a":01}"#,
        r#"{"a":1,}"#,
        r#"{"a" 1}"#,
        "{\"a\":\"\u{1}\"}",
        r#"{"a":"\x"}"#,
        r#"{"a":tru}"#,
        r#"{"a":[1 2]}"#,
    ] {
        assert!(!matches(&grammar, bad), "{bad}");
    }
}

#[test]
fn compact_json_uses_reference_separators() {
    let options = JsonOptions {
        any_whitespace: false,
        ..JsonOptions::default()
    };
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"type": "integer"}}}, "required": ["a", "b"]});
    let grammar = json_schema_grammar(&schema, options).unwrap();
    assert!(matches(&grammar, r#"{"a": 1, "b": [1, 2]}"#));
    assert!(!matches(&grammar, r#"{"a":1,"b":[1,2]}"#));
    assert!(!matches(&grammar, r#"{"a": 1,  "b": []}"#));
}

#[test]
fn objects_follow_schema_order_and_required_properties() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 3},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "tags": {"type": "array", "items": {"enum": ["red", "green"]}, "minItems": 1, "maxItems": 2}
        },
        "required": ["age"]
    });
    let grammar = schema_grammar(&schema);
    for good in [
        r#"{"age":0}"#,
        r#"{"name":"abc","age":120}"#,
        r#"{"age":7,"tags":["red"]}"#,
        r#"{"name":"","age":99,"tags":["green","red"]}"#,
    ] {
        assert!(matches(&grammar, good), "{good}");
    }
    for bad in [
        r#"{}"#,
        r#"{"name":"abc"}"#,
        r#"{"age":121}"#,
        r#"{"age":-1}"#,
        r#"{"age":1.5}"#,
        r#"{"name":"abcd","age":1}"#,
        r#"{"age":1,"name":"a"}"#,
        r#"{"age":1,"tags":[]}"#,
        r#"{"age":1,"tags":["red","red","red"]}"#,
        r#"{"age":1,"tags":["blue"]}"#,
        r#"{"age":1,"other":1}"#,
    ] {
        assert!(!matches(&grammar, bad), "{bad}");
    }
}

#[test]
fn additional_properties_follow_declared_ones() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": {"type": "boolean"}
    });
    let grammar = schema_grammar(&schema);
    assert!(matches(&grammar, r#"{"a":1,"x":true,"y":false}"#));
    assert!(matches(&grammar, r#"{"x":true}"#));
    assert!(!matches(&grammar, r#"{"x":1}"#));
    assert!(!matches(&grammar, r#"{"x":true,"a":1}"#));
    // Without properties or additionalProperties, any keys are admitted.
    let open = schema_grammar(&json!({"type": "object"}));
    assert!(matches(&open, r#"{"k":[1,{"z":null}]}"#));
}

#[test]
fn scalar_keywords_are_exact() {
    let cases: &[(Value, &[&str], &[&str])] = &[
        (
            json!({"const": {"a": [1, "x"]}}),
            &[r#"{"a":[1,"x"]}"#],
            &[r#"{"a": [1,"x"]}"#],
        ),
        (
            json!({"enum": ["a", 1, null, true]}),
            &[r#""a""#, "1", "null", "true"],
            &["false", "2"],
        ),
        (
            json!({"type": ["string", "null"]}),
            &[r#""x""#, "null"],
            &["1"],
        ),
        (json!({"type": "boolean"}), &["true", "false"], &["null"]),
        (
            json!({"type": "number"}),
            &["0", "-0.5", "1e10", "12.5E-3", "-0"],
            &["01", ".5", "1.", "+1", "1e"],
        ),
        (
            json!({"type": "integer", "exclusiveMinimum": -3, "maximum": 1000}),
            &["-2", "0", "999", "1000"],
            &["-3", "1001", "-10", "00"],
        ),
        (
            json!({"type": "integer", "minimum": 17}),
            &["17", "99", "100", "123456"],
            &["16", "9", "-1"],
        ),
        (
            json!({"type": "integer", "maximum": -5}),
            &["-5", "-99", "-100000"],
            &["-4", "0", "5"],
        ),
        (
            json!({"type": "number", "minimum": 0, "maximum": 10}),
            &["0", "0.25", "9.999", "10", "10.000"],
            &["-0.1", "10.01", "11", "1e1"],
        ),
        (
            json!({"type": "number", "exclusiveMinimum": 2, "exclusiveMaximum": 3}),
            &["2.5", "2.001"],
            &["2", "2.0", "3", "3.0", "1"],
        ),
        (
            json!({"type": "number", "maximum": -1}),
            &["-1", "-1.0", "-1.5", "-20"],
            &["-0.5", "0", "-1.01e3"],
        ),
        (
            json!({"type": "string", "minLength": 2, "maxLength": 3}),
            &[r#""ab""#, r#""é😀x""#, r#""\n\"""#],
            &[r#""a""#, r#""abcd""#],
        ),
        (
            json!({"type": "string", "pattern": "^[a-c]+-\\d{2}$"}),
            &[r#""abc-12""#],
            &[r#""abd-12""#, r#""abc-1""#],
        ),
        (
            json!({"type": "string", "pattern": "a\"b"}),
            &[r#""a\"b""#],
            &[r#""a"b""#],
        ),
        (
            json!({"type": "string", "format": "date-time"}),
            &[
                r#""2024-02-29T12:00:00Z""#,
                r#""2024-02-29T12:00:00.5+05:30""#,
            ],
            &[r#""2024-13-01T00:00:00Z""#, r#""2024-02-29 12:00:00Z""#],
        ),
        (
            json!({"type": "string", "format": "uuid"}),
            &[r#""123e4567-e89b-12d3-a456-426614174000""#],
            &[r#""123e4567e89b12d3a456426614174000""#],
        ),
        (
            json!({"type": "string", "format": "ipv4"}),
            &[r#""192.168.0.255""#],
            &[r#""256.1.1.1""#, r#""1.2.3""#],
        ),
        (
            json!({"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 1}]}),
            &["5", r#""z""#],
            &[r#""zz""#, "null"],
        ),
        (
            json!({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": false}),
            &[r#"[1,"a"]"#],
            &["[1]", r#"[1,"a",2]"#],
        ),
        (json!({"type": "array", "maxItems": 0}), &["[]"], &["[1]"]),
    ];
    for (schema, good, bad) in cases {
        let grammar = schema_grammar(schema);
        for text in *good {
            assert!(matches(&grammar, text), "{schema} should accept {text}");
        }
        for text in *bad {
            assert!(!matches(&grammar, text), "{schema} should refuse {text}");
        }
    }
}

#[test]
fn recursive_references_compile() {
    let schema = json!({
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}
                },
                "required": ["value"]
            }
        },
        "$ref": "#/$defs/node"
    });
    let grammar = schema_grammar(&schema);
    assert!(matches(
        &grammar,
        r#"{"value":1,"children":[{"value":2},{"value":3,"children":[]}]}"#
    ));
    assert!(!matches(
        &grammar,
        r#"{"value":1,"children":[{"children":[]}]}"#
    ));
}

#[test]
fn invalid_schemas_are_refused() {
    assert!(matches!(
        json_schema_grammar(&json!({"$ref": "#/missing"}), JsonOptions::default()),
        Err(GrammarError::Invalid(_))
    ));
    let vocabulary = adversarial_vocabulary();
    let never = json_schema_grammar(&json!(false), JsonOptions::default()).unwrap();
    assert!(matches!(
        CompiledGrammar::new(&never, vocabulary),
        Err(GrammarError::Invalid(_))
    ));
}

#[test]
fn regex_and_choice_grammars() {
    let phone = regex_grammar(r"^\d{3}-\d{4}$").unwrap();
    assert!(matches(&phone, "555-1234"));
    assert!(!matches(&phone, "555-123"));
    let words = regex_grammar(r"(?:ab|cd)+[^x\n]?|\w\s\.").unwrap();
    for good in ["ab", "abcdab", "cdé", "a .", "_\t."] {
        assert!(matches(&words, good), "{good}");
    }
    for bad in ["", "abx", "a  .", "ab\n"] {
        assert!(!matches(&words, bad), "{bad}");
    }
    let braces = regex_grammar(r"a{2,}b{,3}").unwrap();
    assert!(matches(&braces, "aab{,3}"));
    for pattern in [r"(a)\1", r"(?=a)", r"a^b", r"\p{L}", r"a++", "*a", "(a"] {
        assert!(regex_grammar(pattern).is_err(), "{pattern}");
    }
    let choice =
        choice_grammar(&["yes".to_owned(), "no".to_owned(), "yes please".to_owned()]).unwrap();
    for good in ["yes", "no", "yes please"] {
        assert!(matches(&choice, good));
    }
    assert!(!matches(&choice, "yes pleas"));
    assert!(choice_grammar(&[]).is_err());
}

/// Runs in a child process with a 2 MiB stack, the size of a Tokio blocking
/// thread, so a stack overflow fails this test instead of aborting the runner.
#[test]
fn regex_nesting_is_bounded_before_compiling() {
    const CHILD: &str = "CHOPINFER_GRAMMAR_REGEX_NESTING_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "regex_nesting_is_bounded_before_compiling"])
            .arg("--nocapture")
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success(), "child exited with {status}");
        return;
    }
    let child = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            let stacked = format!("a{}", "{0,1}".repeat(20_000));
            assert!(regex_grammar(&stacked).is_err());
            for pattern in ["a{2}{3}", "a**", "a+??", "a?*", "(?:a{0,1}){0,1}{0,1}"] {
                assert!(regex_grammar(pattern).is_err(), "{pattern}");
            }
            for pattern in ["a{2}", "a*?", "a+?b", "(?:a?)?", "a{0,1}{,3}"] {
                assert!(regex_grammar(pattern).is_ok(), "{pattern}");
            }
            let deepest = format!("{}a{}", "(?:b|".repeat(64), "){0,1}".repeat(64));
            let grammar = regex_grammar(&deepest).unwrap();
            assert_eq!(grammar.clone(), grammar);
            let vocabulary = adversarial_vocabulary();
            let mut matcher = GrammarMatcher::new(compile(&grammar, &vocabulary));
            assert!(matcher.accept_bytes(b"a").is_ok() && matcher.is_complete());
            drop(matcher);
            drop(grammar);
            assert!(regex_grammar(&format!("({deepest})")).is_err());
        })
        .unwrap();
    child.join().unwrap();
}

#[test]
fn masks_equal_brute_force_on_random_walks() {
    let vocabulary = adversarial_vocabulary();
    let schemas = [
        json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"enum": ["red", "green", "blue"]}}}, "required": ["name"]}),
        json!({"type": "array", "items": {"type": "number"}, "maxItems": 3}),
        json!({"type": "object", "additionalProperties": {"type": "object"}}),
        json!({"$defs": {"t": {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"$ref": "#/$defs/t"}]}}}, "$ref": "#/$defs/t"}),
    ];
    let mut grammars: Vec<(Grammar, Option<Value>)> = schemas
        .iter()
        .map(|schema| (schema_grammar(schema), Some(schema.clone())))
        .collect();
    grammars.push((
        json_object_grammar(JsonOptions::default()),
        Some(json!({"type": "object"})),
    ));
    grammars.push((regex_grammar(r"(ab|a)*c?[é€😀]{0,2}").unwrap(), None));
    for (index, (grammar, schema)) in grammars.iter().enumerate() {
        let compiled = compile(grammar, &vocabulary);
        let mut rng = Rng::new(index as u64 + 1);
        for _ in 0..40 {
            let (text, finished) = random_walk(&compiled, &mut rng, 60, true);
            if !finished {
                continue;
            }
            if let Some(schema) = schema {
                let Some(value) = parse_json(&text) else {
                    continue;
                };
                assert!(validates(&value, schema, schema), "{value} for {schema}");
            } else {
                assert!(std::str::from_utf8(&text).is_ok());
            }
        }
    }
}

#[test]
fn every_generated_output_validates() {
    let vocabulary = adversarial_vocabulary();
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": -50, "maximum": 50},
            "label": {"type": "string", "maxLength": 4},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "kind": {"enum": ["red", "green", "blue", null]},
            "flag": {"type": "boolean"},
            "nested": {"type": "object", "properties": {"x": {"type": "null"}}, "required": ["x"]},
            "list": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}
        },
        "required": ["id", "kind", "list"],
        "additionalProperties": false
    });
    let compiled = compile(&schema_grammar(&schema), &vocabulary);
    let mut rng = Rng::new(7);
    let mut finished = 0;
    for _ in 0..300 {
        let (text, done) = random_walk(&compiled, &mut rng, 200, false);
        if done {
            finished += 1;
            let value = parse_json(&text).expect("bounded numbers parse");
            assert!(validates(&value, &schema, &schema), "{value}");
        }
    }
    assert!(finished > 100, "only {finished} walks finished");
}

/// Keywords beside `enum` and `const` narrow their values, checked by an
/// independent validator. Unsupported intersections are refused.
#[test]
fn literals_and_supported_combinators_enforce_schemas() {
    let narrowed: &[(Value, &[&str], &[&str])] = &[
        (
            json!({"type": "integer", "enum": ["oops", 1, 2.5]}),
            &["1"],
            &[r#""oops""#, "2.5"],
        ),
        (json!({"const": 2, "enum": [1, 2]}), &["2"], &["1"]),
    ];
    for (schema, good, bad) in narrowed {
        let grammar = schema_grammar(schema);
        for text in *good {
            assert!(matches(&grammar, text), "{schema} admits {text}");
            let value = serde_json::from_str(text).unwrap();
            assert!(
                validates(&value, schema, schema),
                "{schema} validates {text}"
            );
        }
        for text in *bad {
            assert!(!matches(&grammar, text), "{schema} refuses {text}");
        }
    }
    assert!(
        runtime_grammar::json_schema_grammar(
            &json!({"type":"string","maxLength":2,"const":"abc"}),
            JsonOptions::default(),
        )
        .is_err()
    );
    let single = schema_grammar(&json!({"allOf":[{"type":"integer"}]}));
    assert!(matches(&single, "1"));
    assert!(!matches(&single, r#""x""#));

    let vocabulary = adversarial_vocabulary();
    let schema = json!({
        "type": "object",
        "properties": {
            "shape": {"anyOf": [
                {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}},
                 "required": ["kind"]},
                {"type": "object", "properties": {"kind": {"enum": ["b", "c"]}, "y": {"type": "boolean"}},
                 "required": ["kind"]}
            ]},
            "size": {"type": "integer", "enum": [1, "two", 3], "maximum": 2},
            "tag": {"anyOf": [{"type": "null"}, {"maxLength": 2}]}
        },
        "required": ["shape", "size", "tag"],
        "additionalProperties": false
    });
    let compiled = compile(&schema_grammar(&schema), &vocabulary);
    let mut rng = Rng::new(11);
    let mut finished = 0;
    for _ in 0..300 {
        let (text, done) = random_walk(&compiled, &mut rng, 200, false);
        if done {
            finished += 1;
            let value = parse_json(&text).expect("the walk is JSON");
            assert!(validates(&value, &schema, &schema), "{value}");
        }
    }
    assert!(finished > 100, "only {finished} walks finished");
}

#[test]
fn stop_tokens_are_admitted_only_at_the_end() {
    let vocabulary = adversarial_vocabulary();
    let compiled = compile(&choice_grammar(&["ab".to_owned()]).unwrap(), &vocabulary);
    let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![STOP]);
    let mut mask = vec![0; vocabulary.mask_words()];
    assert!(state.fill_mask(&mut mask).unwrap());
    assert!(!bit(&mask, STOP));
    assert!(state.accept(STOP).is_err());
    let a = token_id(&vocabulary, b"a");
    let b = token_id(&vocabulary, b"b");
    state.accept(a).unwrap();
    state.accept(b).unwrap();
    state.fill_mask(&mut mask).unwrap();
    assert_eq!(allowed(&mask, vocabulary.size()), vec![STOP]);
    state.accept(STOP).unwrap();
    assert!(state.is_terminated());
    // After termination every row is unconstrained, as in the reference.
    assert!(!state.fill_mask(&mut mask).unwrap());
    // Without stop tokens a finished grammar stops constraining.
    let mut state = ConstraintState::new(compiled, None, Vec::new());
    state.accept(a).unwrap();
    state.accept(b).unwrap();
    assert!(!state.is_active());
}

/// A prefix the prompt ends with advances the constraint, so the rows
/// continue after it; a gated constraint, which begins where its gate
/// opens, takes none.
#[test]
fn a_prefix_advances_the_constraint() {
    let vocabulary = adversarial_vocabulary();
    let compiled = compile(&choice_grammar(&["ab".to_owned()]).unwrap(), &vocabulary);
    let state = ConstraintState::new(Arc::clone(&compiled), None, vec![STOP])
        .with_prefix(b"a")
        .unwrap();
    let mut mask = vec![0; vocabulary.mask_words()];
    state.fill_mask(&mut mask).unwrap();
    let b = token_id(&vocabulary, b"b");
    assert!(bit(&mask, b));
    assert!(!bit(&mask, token_id(&vocabulary, b"a")));
    assert!(
        ConstraintState::new(Arc::clone(&compiled), None, vec![STOP])
            .with_prefix(b"x")
            .is_err()
    );
    let gate = token_id(&vocabulary, b"xyz");
    assert!(
        ConstraintState::new(compiled, Some(vec![gate]), vec![STOP])
            .with_prefix(b"a")
            .is_err()
    );
}

fn token_id(vocabulary: &TokenVocabulary, bytes: &[u8]) -> u32 {
    (0..vocabulary.size() as u32)
        .find(|&token| vocabulary.ordinary_bytes(token) == Some(bytes))
        .unwrap()
}

#[test]
fn rollback_restores_matcher_gate_and_termination() {
    let vocabulary = adversarial_vocabulary();
    let compiled = compile(&choice_grammar(&["ab".to_owned()]).unwrap(), &vocabulary);
    let gate = token_id(&vocabulary, b"xyz");
    let mut state = ConstraintState::new(compiled, Some(vec![gate]), vec![STOP]);
    let mut before = vec![0; vocabulary.mask_words()];
    assert!(!state.fill_mask(&mut before).unwrap());
    let a = token_id(&vocabulary, b"a");
    // Before the gate any token is committed without constraint.
    state.accept(a).unwrap();
    state.accept(gate).unwrap();
    assert!(state.fill_mask(&mut before).unwrap());
    state.accept(a).unwrap();
    state.accept(token_id(&vocabulary, b"b")).unwrap();
    state.accept(STOP).unwrap();
    assert!(state.is_terminated());
    state.rollback(3).unwrap();
    let mut after = vec![0; vocabulary.mask_words()];
    assert!(state.fill_mask(&mut after).unwrap());
    assert_eq!(before, after);
    state.rollback(2).unwrap();
    assert!(!state.fill_mask(&mut after).unwrap());
    assert!(state.rollback(1).is_err());
}

#[test]
fn a_gate_token_the_grammar_expects_is_consumed() {
    let vocabulary = adversarial_vocabulary();
    let grammar = Grammar::from_expr(Expr::seq([Expr::literal("<call>"), Expr::literal("x")]))
        .with_special_tokens([b"<call>".to_vec()]);
    let compiled = compile(&grammar, &vocabulary);
    let mut mask = vec![0; vocabulary.mask_words()];
    let state = ConstraintState::new(Arc::clone(&compiled), None, vec![STOP]);
    state.fill_mask(&mut mask).unwrap();
    // The special token and its ordinary spelling both start the call.
    assert!(bit(&mask, SPECIAL_OPEN));
    assert!(bit(&mask, token_id(&vocabulary, b"<call")));
    let mut gated = ConstraintState::new(compiled, Some(vec![SPECIAL_OPEN]), vec![STOP]);
    gated.accept(SPECIAL_OPEN).unwrap();
    gated.fill_mask(&mut mask).unwrap();
    assert_eq!(
        allowed(&mask, vocabulary.size()),
        vec![token_id(&vocabulary, b"x")]
    );
    // Special tokens a grammar does not name never match.
    let plain = compile(&choice_grammar(&["</s>".to_owned()]).unwrap(), &vocabulary);
    let state = ConstraintState::new(plain, None, vec![]);
    state.fill_mask(&mut mask).unwrap();
    assert!(!bit(&mask, STOP));
}

#[test]
fn tool_structures_build_from_markers_and_schemas() {
    // The shape a frontend builds: markers around a named function whose
    // string parameter is free text up to the close marker.
    let parameters = json!({"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}}, "required": ["city"]});
    let mut builder = GrammarBuilder::new();
    let mut compiler = SchemaCompiler::new(
        &mut builder,
        &parameters,
        JsonOptions {
            any_whitespace: false,
            ..JsonOptions::default()
        },
    );
    let days = compiler.schema(&parameters["properties"]["days"]).unwrap();
    let parameter = |name: &str, value: Expr| {
        Expr::seq([
            Expr::literal("<p="),
            Expr::literal(name),
            Expr::literal(">"),
            value,
            Expr::literal("</p>"),
        ])
    };
    let body = Expr::OrderedSubset {
        items: vec![
            (
                parameter("city", Expr::TextExcluding(vec![b"</p>".to_vec()])),
                true,
            ),
            (parameter("days", days), false),
        ],
        tail: None,
        separator: Box::new(Expr::Empty),
    };
    let call = Expr::seq([Expr::literal("<call>"), body, Expr::literal("</call>")]);
    let root = builder.rule(call.plus());
    let grammar = builder
        .finish(root)
        .unwrap()
        .with_special_tokens([b"<call>".to_vec()]);
    assert!(matches(&grammar, "<call><p=city>Paris </ p></p></call>"));
    assert!(matches(
        &grammar,
        "<call><p=city>a</p><p=days>3</p></call><call><p=city></p></call>"
    ));
    assert!(!matches(&grammar, "<call><p=days>3</p></call>"));
    assert!(!matches(
        &grammar,
        "<call><p=city>a</p><p=days>0</p></call>"
    ));
    assert!(!matches(&grammar, "<call><p=city>a</p></p></call>"));
}

#[test]
fn precompute_classifies_every_position_a_walk_reaches() {
    let vocabulary = adversarial_vocabulary();
    let grammars = [
        schema_grammar(&json!({"type": "array", "items": {"type": "integer"}})),
        schema_grammar(&json!({"type": "object", "properties": {
            "name": {"type": "string", "maxLength": 12},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "extra": {"type": "object"}
        }, "required": ["name"]})),
        schema_grammar(
            &json!({"$defs": {"t": {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"$ref": "#/$defs/t"}]}}}, "$ref": "#/$defs/t"}),
        ),
        json_object_grammar(JsonOptions::default()),
        json_object_grammar(JsonOptions {
            any_whitespace: false,
            ..JsonOptions::default()
        }),
    ];
    for (index, grammar) in grammars.iter().enumerate() {
        let compiled = compile(grammar, &vocabulary);
        let positions = compiled.precompute(4).unwrap();
        assert!(positions > 0, "grammar {index}");
        assert_eq!(compiled.cached_positions(), positions);
        // Walks meet only positions that were classified ahead, and the
        // masks still equal a token-by-token check.
        let mut rng = Rng::new(index as u64 + 11);
        for _ in 0..30 {
            random_walk(&compiled, &mut rng, 50, true);
        }
        assert_eq!(compiled.cached_positions(), positions, "grammar {index}");
        let (classes, _) = compiled.cached_classes();
        assert!(classes <= positions);
    }
}

/// A grammar's retained bytes cover its automaton before any
/// classification, and grow by at least the masks precompute caches.
#[test]
fn retained_bytes_count_the_automaton_and_cached_classifications() {
    let vocabulary = adversarial_vocabulary();
    let grammar = schema_grammar(&json!({"type": "object", "properties": {
        "name": {"type": "string", "maxLength": 12},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
    }, "required": ["name"]}));
    let compiled = compile(&grammar, &vocabulary);
    let before = compiled.retained_bytes();
    assert!(before > 0);
    compiled.precompute(2).unwrap();
    let (_, masks) = compiled.cached_classes();
    assert!(masks > 0);
    assert!(compiled.retained_bytes() >= before + masks);
}

/// A first-use classification that exhausts its budget can still grow the
/// scanner the grammar keeps; the growth observer hears of it.
#[test]
fn a_refused_lazy_classification_still_reports_growth() {
    let schema = json!({"$defs": {
        "a": {"type": "array", "items": {"$ref": "#/$defs/u"}},
        "b": {"type": "array", "items": {"$ref": "#/$defs/u"}, "minItems": 1},
        "u": {"anyOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}]}
    }, "$ref": "#/$defs/u"});
    let grammar = schema_grammar(&schema);
    let mut tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect::<Vec<_>>();
    tokens.push(VocabularyToken {
        bytes: vec![b'['; 32],
        special: false,
    });
    let vocabulary = Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap());
    let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
    let notices = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let heard = Arc::clone(&notices);
    assert!(compiled.observe_growth(Box::new(move || {
        heard.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    })));
    let before = compiled.retained_bytes();
    let refused = GrammarMatcher::new(Arc::clone(&compiled))
        .fill_token_bitmask(&mut vec![0; vocabulary.mask_words()]);
    assert!(
        matches!(refused, Err(GrammarError::TooLarge(_))),
        "{refused:?}"
    );
    assert!(compiled.retained_bytes() > before);
    assert!(notices.load(std::sync::atomic::Ordering::SeqCst) > 0);
}

#[test]
fn the_vocabulary_refuses_an_undersized_width() {
    let tokens = vec![
        VocabularyToken {
            bytes: b"a".to_vec(),
            special: false,
        };
        3
    ];
    assert!(TokenVocabulary::new(2, tokens).is_err());
}

#[test]
fn masks_match_brute_force_for_deep_nesting() {
    let vocabulary = adversarial_vocabulary();
    let compiled = compile(&json_object_grammar(JsonOptions::default()), &vocabulary);
    let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
    let mut mask = vec![0; vocabulary.mask_words()];
    for piece in [
        "{", "\"a\"", ":", "[", "[", "{", "\"b\"", ":", "1", "}", "]", ",", "\"x", "\"", "]",
    ] {
        matcher.accept_bytes(piece.as_bytes()).unwrap();
        matcher.fill_token_bitmask(&mut mask).unwrap();
        assert_eq!(
            allowed(&mask, vocabulary.size()),
            brute_force(&matcher, &vocabulary),
            "after {piece}"
        );
    }
}

// The synthetic model wants to finish {}, but whitespace has virtually all
// probability mass whenever the grammar permits it (temperature 1, top_p .95).
fn whitespace_sample(grammar: &Grammar, seed: u64) -> (Vec<u8>, bool) {
    let vocabulary = adversarial_vocabulary();
    let mut state = ConstraintState::new(compile(grammar, &vocabulary), None, vec![STOP]);
    let mut rng = Rng::new(seed);
    let mut text = Vec::new();
    let mut mask = vec![0; vocabulary.mask_words()];
    for _ in 0..4096 {
        state.fill_mask(&mut mask).unwrap();
        let preferred: Vec<_> = [b' ', b'\n', b'\t']
            .into_iter()
            .map(|byte| u32::from(byte) + 2)
            .filter(|&token| bit(&mask, token))
            .collect();
        let token = if !preferred.is_empty() {
            // Equal whitespace logits dominate the structural logits by 100,
            // so nucleus sampling retains just these equiprobable tokens.
            preferred[rng.below(preferred.len())]
        } else if bit(&mask, STOP) {
            STOP
        } else if bit(&mask, u32::from(b'}') + 2) {
            u32::from(b'}') + 2
        } else {
            u32::from(b'{') + 2
        };
        state.accept(token).unwrap();
        if token == STOP {
            return (text, true);
        }
        text.extend(common::token_text(&vocabulary, token));
    }
    (text, false)
}

#[test]
fn whitespace_preferring_sampling_finishes_valid_json() {
    for grammar in [
        json_object_grammar(JsonOptions::default()),
        schema_grammar(&json!({"type":"object","properties":{},"additionalProperties":false})),
    ] {
        for seed in 0..8 {
            let (text, stopped) = whitespace_sample(&grammar, seed);
            assert!(stopped, "whitespace exhausted the 4096-token budget");
            assert_eq!(serde_json::from_slice::<Value>(&text).unwrap(), json!({}));
            assert!(text.len() <= 259);
        }
    }
}

#[test]
fn whitespace_runs_are_bounded_even_in_empty_containers() {
    let vocabulary = adversarial_vocabulary();
    for grammar in [
        json_object_grammar(JsonOptions::default()),
        schema_grammar(&json!(true)),
    ] {
        let compiled = compile(&grammar, &vocabulary);
        for (prefix, suffix) in [
            ("{", "}"),
            ("{\"a\":[", "]}"),
            ("{\"a\":0", "}"),
            ("{\"a\"", ":0}"),
            ("{\"a\":", "0}"),
            ("{\"a\":0,", "\"b\":1}"),
        ] {
            for count in [0, 1, 254, 255, 256, 257] {
                let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
                matcher.accept_bytes(prefix.as_bytes()).unwrap();
                matcher.accept_bytes(" ".repeat(count).as_bytes()).unwrap();
                let mut mask = vec![0; vocabulary.mask_words()];
                matcher.fill_token_bitmask(&mut mask).unwrap();
                assert_eq!(
                    allowed(&mask, vocabulary.size()),
                    brute_force(&matcher, &vocabulary)
                );
                assert_eq!(
                    matcher.accepts_token(u32::from(b' ') + 2).unwrap(),
                    count < 257,
                    "{prefix}"
                );
                matcher.accept_bytes(suffix.as_bytes()).unwrap();
                assert!(matcher.is_complete());
            }
        }
    }
}

#[test]
fn unbounded_whitespace_restores_reference_sampling_behavior() {
    let options = JsonOptions {
        max_whitespace_cnt: None,
        ..JsonOptions::default()
    };
    for grammar in [
        json_object_grammar(options),
        json_schema_grammar(&json!({"type":"object"}), options).unwrap(),
    ] {
        let (_, stopped) = whitespace_sample(&grammar, 7);
        assert!(!stopped);
        assert!(matches(&grammar, &format!("{{{}}}", " \n\t".repeat(4096))));
        assert!(!matches(&grammar, "{\r}"));
    }
}

#[test]
fn pretty_printed_deep_json_and_string_spaces_remain_valid() {
    let vocabulary = adversarial_vocabulary();
    for grammar in [
        json_object_grammar(JsonOptions::default()),
        schema_grammar(&json!(true)),
    ] {
        let compiled = compile(&grammar, &vocabulary);
        // Deep four-space indentation, including CRLF.
        let mut value = json!({"text": " ".repeat(1024)});
        for _ in 1..63 {
            value = json!({"nested": value});
        }
        let text = serde_json::to_string_pretty(&value).unwrap();
        // Expand serde_json's two-space indentation without touching strings.
        let text = text
            .lines()
            .map(|line| {
                let indent = line.len() - line.trim_start().len();
                format!("{}{}", " ".repeat(indent), line)
            })
            .collect::<Vec<_>>()
            .join("\r\n");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
        let mut matcher = GrammarMatcher::new(compiled);
        matcher.accept_bytes(text.as_bytes()).unwrap();
        assert!(matcher.is_complete());
    }
}

#[test]
fn whitespace_boundary_survives_forks_and_rollback() {
    let vocabulary = adversarial_vocabulary();
    let grammar = compile(&json_object_grammar(JsonOptions::default()), &vocabulary);
    let mut state = ConstraintState::new(grammar, None, vec![STOP]);
    state.accept(u32::from(b'{') + 2).unwrap();
    for _ in 0..257 {
        state.accept(u32::from(b' ') + 2).unwrap();
    }
    let mut fork = state.fork();
    assert!(!fork.allows(u32::from(b' ') + 2).unwrap());
    fork.accept(u32::from(b'}') + 2).unwrap();
    fork.accept(STOP).unwrap();
    assert!(fork.is_terminated());
    state.rollback(1).unwrap();
    assert!(state.allows(u32::from(b' ') + 2).unwrap());
    state.accept(u32::from(b'\n') + 2).unwrap();
    assert!(!state.allows(u32::from(b' ') + 2).unwrap());
}

/// Long whitespace budgets share classification work, while every remaining
/// budget and mixed-token boundary still agrees with the byte matcher.
#[test]
fn whitespace_classification_is_shared_and_exact_at_every_budget() {
    for mixed in [false, true] {
        let mut tokens: Vec<_> = (0..=255u8)
            .map(|byte| VocabularyToken {
                bytes: vec![byte],
                special: false,
            })
            .collect();
        if mixed {
            for text in [
                "    ",
                "       }",
                "\"        }",
                "1        ]}",
                "\r\n\t }",
                "{}",
            ] {
                tokens.push(VocabularyToken {
                    bytes: text.as_bytes().to_vec(),
                    special: false,
                });
            }
        }
        let vocabulary = Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap());
        let compiled = compile(&json_object_grammar(JsonOptions::default()), &vocabulary);
        let positions = compiled.precompute(4).unwrap();
        assert!(
            (1..2048).contains(&positions),
            "classified {positions} positions"
        );
        for prefix in ["{", "{\"a\":1", "{\"a\":[", "{\"a\":[1", "{\"a\":{\"b\":1"] {
            let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
            matcher.accept_bytes(prefix.as_bytes()).unwrap();
            for remaining in (0..=257).rev() {
                let mut mask = vec![0; vocabulary.mask_words()];
                matcher.fill_token_bitmask(&mut mask).unwrap();
                for token in 0..vocabulary.size() as u32 {
                    assert_eq!(
                        bit(&mask, token),
                        matcher.accepts_token(token).unwrap(),
                        "{prefix} budget {remaining}, token {token}"
                    );
                }
                if remaining > 0 {
                    matcher.accept_bytes(b" ").unwrap();
                }
            }
            assert!(matcher.accept_bytes(b" ").is_err());
        }
        assert_eq!(compiled.cached_positions(), positions);
    }
}

#[test]
fn shared_top_frame_scans_preserve_the_recursion_limit() {
    let mut builder = GrammarBuilder::new();
    let nested = builder.declare();
    builder.define(
        nested,
        Expr::alt([
            Expr::literal("x"),
            Expr::seq([Expr::literal("("), Expr::Rule(nested), Expr::literal(")")]),
        ]),
    );
    let root = builder.rule(Expr::seq([Expr::Rule(nested), Expr::literal("!")]));
    let grammar = builder.finish(root).unwrap();
    let tokens = [
        "(".repeat(2047),
        format!("{}x", "(".repeat(2046)),
        "x".into(),
        "x!".into(),
    ]
    .into_iter()
    .map(|text| VocabularyToken {
        bytes: text.into_bytes(),
        special: false,
    })
    .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(4, tokens).unwrap());
    let matcher = GrammarMatcher::new(compile(&grammar, &vocabulary));
    let mut mask = vec![0; vocabulary.mask_words()];
    assert!(matches!(
        matcher.fill_token_bitmask(&mut mask),
        Err(GrammarError::TooLarge(_))
    ));
    assert!(mask.iter().all(|word| *word == 0));
    assert!(matches!(
        matcher.accepts_token(2),
        Err(GrammarError::TooLarge(_))
    ));
}

fn depth_budget_fixture(wrapped: bool) -> Arc<CompiledGrammar> {
    let mut builder = GrammarBuilder::new();
    let nested = builder.declare();
    let body = Expr::alt([
        Expr::literal("x"),
        Expr::seq([Expr::literal("("), Expr::Rule(nested), Expr::literal(")")]),
    ]);
    if wrapped {
        // Two calls per opening byte, including a call before any byte.
        let wrapper = builder.rule(body);
        builder.define(nested, Expr::Rule(wrapper));
    } else {
        builder.define(nested, body);
    }
    let root = builder.rule(Expr::seq([Expr::Rule(nested), Expr::literal("!")]));
    let grammar = builder.finish(root).unwrap();
    let tokens = [
        "(".into(),
        "x".into(),
        ")".into(),
        "!".into(),
        "(".repeat(2046),
        "(".repeat(1023),
        "(".repeat(2040),
        "((x))".into(),
        ")((".into(),
        "x)!".into(),
    ]
    .into_iter()
    .map(|text: String| VocabularyToken {
        bytes: text.into_bytes(),
        special: false,
    })
    .collect::<Vec<_>>();
    let vocabulary = Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap());
    compile(&grammar, &vocabulary)
}

#[test]
fn masks_preserve_remaining_depth_after_prefixes() {
    for wrapped in [false, true] {
        let compiled = depth_budget_fixture(wrapped);
        if wrapped {
            assert!(matches!(
                compiled.precompute(2),
                Err(GrammarError::TooLarge(_))
            ));
        } else {
            compiled.precompute(2).unwrap();
        }
        for count in [1, 2, 5, if wrapped { 1020 } else { 2044 }] {
            let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
            matcher.accept_bytes("(".repeat(count).as_bytes()).unwrap();
            let mut mask = vec![0; compiled.vocabulary().mask_words()];
            assert!(matches!(
                matcher.fill_token_bitmask(&mut mask),
                Err(GrammarError::TooLarge(_))
            ));
            assert!(mask.iter().all(|word| *word == 0));
            // The cheap token "x" must not become the surviving distribution.
            assert!(matches!(
                matcher.accepts_token(1),
                Err(GrammarError::TooLarge(_))
            ));
        }
    }
}

#[test]
fn remaining_depth_masks_survive_forks_and_rollback() {
    for wrapped in [false, true] {
        let compiled = depth_budget_fixture(wrapped);
        let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![]);
        let prefix = if wrapped { 1019 } else { 2040 };
        for _ in 0..prefix {
            state.accept(0).unwrap();
        }
        let original = state.fork();
        let mut fork = state.fork();
        for _ in 0..3 {
            fork.accept(0).unwrap();
        }
        let mut mask = vec![u32::MAX; compiled.vocabulary().mask_words()];
        assert!(matches!(
            fork.fill_mask(&mut mask),
            Err(GrammarError::TooLarge(_))
        ));
        assert!(mask.iter().all(|word| *word == 0));
        // Exhaustion is terminal for the request, including existing forks
        // and checkpoints: rollback cannot turn it into unconstrained output.
        fork.rollback(3).unwrap();
        assert_eq!(fork, original);
        for current in [&mut fork, &mut state] {
            assert!(current.is_active());
            assert!(matches!(
                current.fill_mask(&mut mask),
                Err(GrammarError::TooLarge(_))
            ));
            assert!(matches!(current.allows(1), Err(GrammarError::TooLarge(_))));
            assert!(matches!(current.accept(1), Err(GrammarError::TooLarge(_))));
        }
    }
}
