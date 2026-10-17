//! Forced calls with literal values: a string enumeration or constant value
//! the tool-choice grammar writes as it is, which the served parser must
//! read back whole. Also forced calls whose parameter names an additional
//! property could share with a declared one, which must read back as the
//! grammar typed them.

use std::sync::Arc;

use runtime_grammar::{CompiledGrammar, Grammar, GrammarMatcher, TokenVocabulary, VocabularyToken};
use serde_json::{Map, Value, json};

use crate::chat::PromptModeV1;
use crate::descriptor::FormatDescriptorV1;
use crate::format::{FormatEventV1, FormatInputV1, FormatParserOptionsV1};
use crate::format_decode::{DeclaredParserV1, completion_calls_v1};
use crate::request::{FrontendToolChoiceV1, FrontendToolV1};
use crate::tool_grammar::tool_constraint_v1;

/// One call to `f` with the parameter `x` holding `value`, written with the
/// declaration's markers and wrappers exactly, whitespace included, as the
/// tool-choice grammar writes it.
#[must_use]
pub fn forced_call(descriptor: &FormatDescriptorV1, value: &str) -> String {
    forced_call_with(descriptor, &[("x", value)])
}

/// One call to `f` with `arguments`, names and value texts in order,
/// written as [`forced_call`] writes its one argument.
#[must_use]
pub fn forced_call_with(descriptor: &FormatDescriptorV1, arguments: &[(&str, &str)]) -> String {
    let parser = &descriptor.parser;
    let mut text = [
        parser.tool_call_open.as_str(),
        &parser.function_open,
        "f",
        &parser.function_name_close,
    ]
    .concat();
    for (name, value) in arguments {
        text.extend([
            parser.parameter_open.as_str(),
            name,
            &parser.parameter_name_close,
            &parser.parameter_value_open,
            value,
            &parser.parameter_close,
        ]);
    }
    text.extend([parser.function_close.as_str(), &parser.tool_call_close]);
    text
}

/// Whether the grammar `tool_choice: required` imposes on `tools` admits
/// `text` whole, over a vocabulary of one token per byte; `None` when the
/// tools are refused.
#[must_use]
pub fn required_admits(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
    text: &str,
) -> Option<bool> {
    required_grammar(descriptor, tools).map(|compiled| admits(&compiled, text))
}

/// The grammar `tool_choice: required` imposes on `tools`, compiled over a
/// vocabulary of one token per byte; `None` when the tools are refused.
fn required_grammar(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
) -> Option<Arc<CompiledGrammar>> {
    let grammar = tool_constraint_v1(descriptor, tools, &FrontendToolChoiceV1::Required)
        .ok()?
        .expect("a required tool choice constrains");
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).expect("a byte vocabulary"));
    Some(Arc::new(
        CompiledGrammar::new(&grammar, vocabulary).expect("the grammar compiles"),
    ))
}

fn admits(compiled: &Arc<CompiledGrammar>, text: &str) -> bool {
    let mut matcher = GrammarMatcher::new(Arc::clone(compiled));
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

fn tool(parameters: Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value(parameters).expect("a schema"),
        parameters_present: true,
        strict: None,
    }
}

/// Checks a declaration's forced literals: under `tool_choice: required`,
/// an enumeration, a constant and an additional-property enumeration value
/// that spells no marker builds a grammar that writes the call, which the
/// served parser reads back as that value, under `string`, the `str` alias
/// and an unknown type name; a value that spells the parameter close is
/// refused. Then [`assert_admitted_names_read_back`].
///
/// # Panics
///
/// When any of these does not hold.
pub fn assert_forced_literals_read_back(descriptor: &FormatDescriptorV1) {
    let parser = &descriptor.parser;
    let served = Arc::new(descriptor.clone());
    for kind in ["string", "str", "unknown-type"] {
        for value in ["plain", "\"quoted\"", "two words", ""] {
            for parameters in [
                json!({"type": "object", "properties": {"x": {"type": kind, "enum": [value]}},
                       "required": ["x"], "additionalProperties": false}),
                json!({"type": "object", "properties": {"x": {"type": kind, "const": value}},
                       "required": ["x"]}),
                json!({"type": "object", "required": ["x"],
                       "additionalProperties": {"type": kind, "enum": [value]}}),
            ] {
                let tools = [tool(parameters)];
                let text = forced_call(descriptor, value);
                let label = format!("{kind} {value:?} {}", tools[0].parameters.to_json_value());
                assert_eq!(
                    required_admits(descriptor, &tools, &text),
                    Some(true),
                    "{label}: the grammar refuses or does not write the call"
                );
                let calls = completion_calls_v1(&served, &text, &tools)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                let read: Vec<(String, Value)> = calls
                    .into_iter()
                    .map(|call| (call.name, serde_json::from_str(&call.arguments).unwrap()))
                    .collect();
                assert_eq!(read, [("f".to_owned(), json!({"x": value}))], "{label}");
            }
        }
        let value = format!("a{}b", parser.parameter_close.trim());
        let tools = [tool(
            json!({"type": "object", "properties": {"x": {"type": kind, "enum": [value]}},
                   "required": ["x"]}),
        )];
        assert_eq!(
            required_admits(descriptor, &tools, &forced_call(descriptor, &value)),
            None,
            "{kind} {value:?}: a value that spells the parameter close is offered"
        );
    }
    assert_admitted_names_read_back(descriptor);
}

/// Parameters whose additional or pattern properties could name a declared
/// property, or a required one, with a value its own schema does not type,
/// each with a call of only its declared and required names. The first is
/// the review's reproducer: a tail entry `x` holding `"open` read back as
/// `{}`, since the integer `x` scans the quote as a JSON string.
fn name_cases() -> [(Value, &'static [(&'static str, &'static str)]); 6] {
    [
        (
            json!({"type": "object", "properties": {"x": {"type": "integer"}},
                   "additionalProperties": {"type": "str", "enum": ["\"open"]}}),
            &[("x", "42")],
        ),
        (
            json!({"type": "object", "properties": {"x": {"type": "integer"}},
                   "additionalProperties": {"type": "string"}}),
            &[("x", "42")],
        ),
        (
            json!({"type": "object", "properties": {"x": {"type": "integer"}},
                   "additionalProperties": {"type": "integer"}}),
            &[("x", "42")],
        ),
        (
            json!({"type": "object", "properties": {"x": {"type": "integer"}},
                   "patternProperties": {"^y": {"type": "integer"}}}),
            &[("x", "42")],
        ),
        (
            json!({"type": "object", "properties": {"x": {"type": "string"}},
                   "required": ["y", "y"], "additionalProperties": {"type": "integer"}}),
            &[("y", "7")],
        ),
        (
            json!({"type": "object", "required": ["x"], "additionalProperties": true}),
            &[("x", "abc")],
        ),
    ]
}

/// What the grammar means by the value text `text` of the parameter `name`
/// under `parameters`: the text read by the schema the name resolves to (the
/// declared property; else free text when pattern properties may type it;
/// none when additional properties are false; the additional-property schema
/// when it is one; otherwise free text), or `None` when that schema does not
/// admit the text. Only the schemas of [`name_cases`] are read.
fn spelled(parameters: &Value, name: &str, text: &str) -> Option<Value> {
    let schema = match parameters["properties"].get(name) {
        Some(schema) => Some(schema),
        None if parameters.get("patternProperties").is_some() => None,
        None => match parameters.get("additionalProperties") {
            Some(Value::Bool(false)) => return None,
            Some(schema @ Value::Object(_)) => Some(schema),
            _ => None,
        },
    };
    let Some(schema) = schema else {
        return Some(json!(text));
    };
    if let Some(values) = schema["enum"].as_array() {
        return values.contains(&json!(text)).then(|| json!(text));
    }
    match schema["type"].as_str() {
        Some("integer") => text.parse::<i64>().ok().map(Value::from),
        _ => Some(json!(text)),
    }
}

/// The calls `descriptor`'s parser delivers for `text` and its stop token,
/// pushed in two parts at the byte `split`, or in one part when it is
/// `None`: whole, and when streamed, also as joined deltas.
fn read_calls(
    descriptor: &Arc<FormatDescriptorV1>,
    tools: &[FrontendToolV1],
    text: &str,
    split: Option<usize>,
    stream: bool,
) -> [Vec<(String, Value)>; 2] {
    let vocabulary = &descriptor.vocabulary;
    let stop = vocabulary.stop_token_ids.first().and_then(|id| {
        vocabulary
            .boundary_tokens
            .iter()
            .find(|boundary| boundary.id == *id)
            .map(|boundary| boundary.text.as_str())
    });
    let completion = format!("{text}{}", stop.unwrap_or_default());
    let mut parser = DeclaredParserV1::new(
        Arc::clone(descriptor),
        FormatParserOptionsV1 {
            mode: PromptModeV1::Chat,
            reasoning_open: false,
            extract_tools: true,
            stream_tool_calls: stream,
            raw: false,
            ignore_eos: false,
            engine_tokens: false,
            tool_schemas: tools.to_vec(),
        },
    );
    let bytes = completion.as_bytes();
    let mut events = Vec::new();
    let push = |parser: &mut DeclaredParserV1, completion: &[u8], last: bool| {
        parser
            .push(FormatInputV1 {
                completion,
                held: false,
                last,
                stop_floor: Some(0),
            })
            .unwrap_or_else(|error| panic!("{text:?} split {split:?}: {error}"))
    };
    if let Some(split) = split {
        events = push(&mut parser, &bytes[..split], false);
    }
    events.extend(push(&mut parser, bytes, true));
    let (mut whole, mut streamed) = (Vec::new(), Vec::<(String, String)>::new());
    for event in events {
        match event {
            FormatEventV1::ToolCalls { calls, .. } => {
                whole.extend(calls.into_iter().map(|call| (call.name, call.arguments)));
            }
            FormatEventV1::ToolCallDeltas(deltas) => {
                for delta in deltas {
                    match delta.name {
                        Some(name) => streamed.push((name, delta.arguments)),
                        None => streamed[delta.index].1.push_str(&delta.arguments),
                    }
                }
            }
            _ => {}
        }
    }
    [whole, streamed].map(|calls| {
        calls
            .into_iter()
            .map(|(name, arguments)| {
                let arguments = serde_json::from_str(&arguments)
                    .unwrap_or_else(|error| panic!("{text:?}: {arguments:?}: {error}"));
                (name, arguments)
            })
            .collect()
    })
}

/// Checks parameter names under `tool_choice: required`: over every call of
/// up to two arguments named `x`, `y` or `z` with the value texts `42`, `7`,
/// `"open` or `abc`, under each of [`name_cases`], each call the grammar
/// admits holds only values their names' schemas type, names no declared or
/// required parameter twice, and reads back, whole and at every byte split,
/// delivered whole and streamed, as the arguments it spelled. A name an
/// additional property repeats keeps its last value, as the parser keeps it.
/// Each case also admits a call with only its declared and required names.
///
/// # Panics
///
/// When any of these does not hold.
pub fn assert_admitted_names_read_back(descriptor: &FormatDescriptorV1) {
    let served = Arc::new(descriptor.clone());
    let pairs: Vec<(&str, &str)> = ["x", "y", "z"]
        .into_iter()
        .flat_map(|name| ["42", "7", "\"open", "abc"].map(|text| (name, text)))
        .collect();
    let candidates: Vec<Vec<(&str, &str)>> = std::iter::once(Vec::new())
        .chain(pairs.iter().map(|pair| vec![*pair]))
        .chain(
            pairs
                .iter()
                .flat_map(|first| pairs.iter().map(|second| vec![*first, *second])),
        )
        .collect();
    for (parameters, needed) in name_cases() {
        let tools = [tool(parameters.clone())];
        let compiled = required_grammar(descriptor, &tools)
            .unwrap_or_else(|| panic!("{parameters}: the tools are refused"));
        let declared = |name: &str| {
            parameters["properties"].get(name).is_some()
                || parameters["required"]
                    .as_array()
                    .is_some_and(|required| required.contains(&json!(name)))
        };
        let mut admitted = 0;
        for arguments in &candidates {
            let text = forced_call_with(descriptor, arguments);
            if !admits(&compiled, &text) {
                continue;
            }
            admitted += 1;
            let label = format!("{parameters} admits {text:?}");
            let mut expected = Map::new();
            for (index, (name, value)) in arguments.iter().enumerate() {
                assert!(
                    !(declared(name) && arguments[..index].iter().any(|(seen, _)| seen == name)),
                    "{label}: a declared or required name twice"
                );
                let value = spelled(&parameters, name, value)
                    .unwrap_or_else(|| panic!("{label}: a value {name}'s schema does not type"));
                expected.insert((*name).to_owned(), value);
            }
            let expected = vec![("f".to_owned(), Value::Object(expected))];
            for split in std::iter::once(None).chain((0..=text.len()).map(Some)) {
                for stream in [false, true] {
                    let [whole, streamed] = read_calls(&served, &tools, &text, split, stream);
                    assert_eq!(whole, expected, "{label} split {split:?} stream {stream}");
                    if stream {
                        assert_eq!(streamed, expected, "{label} split {split:?}");
                    }
                }
            }
        }
        assert!(
            admits(&compiled, &forced_call_with(descriptor, needed)),
            "{parameters}: the grammar does not write {needed:?}"
        );
        assert!(admitted > 1, "{parameters}: admits {admitted} calls");
    }
}

/// Checks that a forced call to tools of 100 optional integer properties
/// and string additional properties builds a grammar that compiles within
/// its budgets and crosses the wire, over a vocabulary of one token per byte:
/// one tool whose property names are short (`p000` to `p099`) or as long as
/// the declaration permits and share all but their last three bytes, three
/// such tools, and two tools whose long names differ in their first three
/// bytes. The additional-property names exclude every property name, which
/// must not cost a bounded run of name bytes per excluded prefix, and each
/// tool's arguments must compile once, not once per place a call may stand.
///
/// # Panics
///
/// When a grammar is refused, exceeds a compilation budget or fails to
/// cross the wire.
pub fn assert_long_property_names_compile(descriptor: &FormatDescriptorV1) {
    let longest = descriptor.limits.maximum_name_bytes;
    let pad = "a".repeat(longest - 3);
    // Each case's tool count and the bytes of its names around their index.
    let cases = [
        ("one tool, names p000 to p099", 1, "p", ""),
        ("one tool, long names sharing a prefix", 1, &pad, ""),
        ("three tools, long names sharing a prefix", 3, &pad, ""),
        ("two tools, long names from distinct prefixes", 2, "", &pad),
    ];
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).expect("a byte vocabulary"));
    for (label, count, before, after) in cases {
        let properties: Map<String, Value> = (0..100)
            .map(|index| {
                let name = format!("{before}{index:03}{after}");
                (name, json!({"type": "integer"}))
            })
            .collect();
        let parameters = json!({"type": "object", "properties": properties,
                                "additionalProperties": {"type": "string"}});
        let tools: Vec<FrontendToolV1> = (0..count)
            .map(|index| FrontendToolV1 {
                name: format!("f{index}"),
                ..tool(parameters.clone())
            })
            .collect();
        let grammar = tool_constraint_v1(descriptor, &tools, &FrontendToolChoiceV1::Required)
            .unwrap_or_else(|error| panic!("{label}: {error:?}"))
            .expect("a required tool choice constrains");
        if let Err(error) = CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)) {
            panic!("{label}: {error}");
        }
        if let Err(error) = Grammar::decode_wire(&grammar.encode_wire()) {
            panic!("{label}: the grammar does not cross the wire: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fixtures::{FUNCTION_PARAMETERS, KEY_VALUE};

    /// The function-parameter fixture, whose markers frame each value in a
    /// newline, the same with markers that frame nothing, and the key-value
    /// fixture.
    fn families() -> [FormatDescriptorV1; 3] {
        let framed = FormatDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes())
            .expect("the fixture reads");
        let mut unframed = framed.clone();
        let parser = &mut unframed.parser;
        parser.tool_call_open = "<tool_call>".into();
        parser.function_name_close = ">".into();
        parser.function_close = "</function>".into();
        parser.parameter_name_close = ">".into();
        parser.parameter_close = "</parameter>".into();
        unframed.validate().expect("the declaration validates");
        let key_value = FormatDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes())
            .expect("the fixture reads");
        [framed, unframed, key_value]
    }

    #[test]
    fn admitted_names_read_back_in_every_family() {
        for descriptor in families() {
            assert_admitted_names_read_back(&descriptor);
        }
    }
}
