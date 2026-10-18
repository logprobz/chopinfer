//! The grammar a tool choice imposes, in the declared tool-call format.
//!
//! Calls are written as the parser declaration's template renders them: the
//! call and function markers around the name, then one parameter block per
//! argument, and the declared separator between calls. Parameters follow the
//! schema's order; required ones must appear. A string parameter is raw text
//! up to its close marker, and any other parameter is JSON for its schema,
//! with no surrounding whitespace so the decoder reads it back as that type.
//!
//! The decoder reads calls by the bare markers, without the template's
//! whitespace, so free text avoids the bare markers that would end it early.
//! A grammar without a function element declares its function markers empty,
//! and an empty marker is written as nothing and avoided nowhere. The typed
//! grammar closes a raw value's name with `parameter_name_close` and a JSON
//! value's with `parameter_json_name_close`.

use std::collections::{BTreeMap, BTreeSet};

use runtime_grammar::{
    CharClass, Expr, GRAMMAR_WIRE_NESTING_MAX_V1, Grammar, GrammarBuilder, GrammarError,
    JsonOptions, SchemaCompiler,
};
use serde_json::Value;

use crate::chatml_decode::{
    ParameterSchemaV1, additional_schema_v1, admits_string_v1, call_open_v1,
    literal_value_reads_back_v1,
};
use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::request::{FrontendErrorV1, FrontendToolChoiceV1, FrontendToolV1};
use crate::toolkit::validate_name_v1;

fn refusal(message: impl Into<String>) -> FrontendErrorV1 {
    FrontendErrorV1::new(message, Some("tool_choice"))
}

fn grammar_error(error: &GrammarError) -> FrontendErrorV1 {
    FrontendErrorV1::new(format!("tools: {error}"), Some("tools"))
}

/// The grammar `choice` imposes on calls to `tools`, in the tool-call format
/// `descriptor` declares, or `None` when it imposes nothing.
pub fn tool_constraint_v1(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
    choice: &FrontendToolChoiceV1,
) -> Result<Option<Grammar>, FrontendErrorV1> {
    let forced = match choice {
        FrontendToolChoiceV1::None => return Ok(None),
        FrontendToolChoiceV1::Auto if !tools.iter().any(|tool| tool.strict == Some(true)) => {
            return Ok(None);
        }
        choice => choice,
    };
    let format = &descriptor.parser;
    let offered: Vec<&FrontendToolV1> = match forced {
        FrontendToolChoiceV1::Named(name) => {
            tools.iter().filter(|tool| tool.name == *name).collect()
        }
        _ => tools.iter().collect(),
    };
    if offered.is_empty() {
        return Err(refusal(
            "The tool specified in `tool_choice` does not match any of the specified `tools`",
        ));
    }
    // With more than one call, the call body is one rule: the first call and
    // the calls after it continue to the same state, so they compile to one
    // copy. A rule nests its body a level deeper, and where that takes the
    // grammar past the depth the wire admits, the body is written out in
    // both places, as one call writes it.
    let shared = !matches!(forced, FrontendToolChoiceV1::Named(_))
        || descriptor.parser.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1;
    let mut grammar = calls_v1(descriptor, &offered, forced, shared)?;
    if shared && grammar.compile_depth() > GRAMMAR_WIRE_NESTING_MAX_V1 {
        grammar = calls_v1(descriptor, &offered, forced, false)?;
    }
    // A marker, or the marker without its layout whitespace, may be a single
    // special token; the grammar admits such tokens where their bytes fit.
    let markers = [
        &format.tool_call_open,
        &format.tool_call_close,
        &format.function_open,
        &format.function_close,
        &format.parameter_open,
        &format.parameter_value_open,
        &format.parameter_close,
    ];
    Ok(Some(
        grammar.with_special_tokens(
            markers
                .into_iter()
                .flat_map(|marker| [marker.as_str(), marker.trim()])
                .filter(|marker| !marker.is_empty())
                .map(|marker| marker.as_bytes().to_vec()),
        ),
    ))
}

/// The calls `forced` admits to the `offered` tools, with the call body as
/// one rule when `shared`.
fn calls_v1(
    descriptor: &FormatDescriptorV1,
    offered: &[&FrontendToolV1],
    forced: &FrontendToolChoiceV1,
    shared: bool,
) -> Result<Grammar, FrontendErrorV1> {
    let format = &descriptor.parser;
    let mut builder = GrammarBuilder::new();
    let mut rests = Vec::with_capacity(offered.len());
    for tool in offered {
        rests.push(call_after_open(&mut builder, descriptor, format, tool)?);
    }
    // Everything a call writes after its open marker.
    let rest = Expr::alt(rests);
    let rest = if shared {
        Expr::Rule(builder.rule(rest))
    } else {
        rest
    };
    let open = || Expr::literal(&format.tool_call_open);
    let typed = format.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1;
    // The typed grammar writes every call of a turn as one function element
    // after another inside one call block, as its reference parser reads
    // them; a named choice is the same block of calls to its one tool.
    // Every other grammar writes a call block per call.
    let call = if typed {
        Expr::seq([
            open(),
            rest.clone(),
            Expr::seq([Expr::literal(&format.tool_call_separator), rest.clone()]).star(),
            Expr::literal(&format.tool_call_close),
        ])
    } else {
        Expr::seq([open(), rest.clone()])
    };
    let more_calls = if typed {
        Expr::Empty
    } else {
        Expr::seq([Expr::literal(&format.tool_call_separator), call.clone()]).star()
    };
    // A template that writes a prefix before the answer may start with it.
    let answer_prefix = Expr::literal(&format.answer_prefix).optional();
    let body = match forced {
        FrontendToolChoiceV1::Named(_) if !typed => Expr::seq([answer_prefix, call]),
        FrontendToolChoiceV1::Named(_) | FrontendToolChoiceV1::Required => {
            Expr::seq([answer_prefix, call, more_calls])
        }
        // Free text, then optionally calls. The decoder reads everything from
        // the first call or function marker on as calls, so the text contains
        // neither and, when calls follow, ends at the first call marker.
        _ => {
            let (tool_open, after_open) = call_open_v1(format);
            let function_open = format.function_open.trim();
            let calls = if typed {
                Expr::seq([
                    rest.clone(),
                    Expr::seq([Expr::literal(&format.tool_call_separator), rest]).star(),
                    Expr::literal(&format.tool_call_close),
                ])
            } else {
                Expr::seq([rest, more_calls])
            };
            Expr::alt([
                Expr::TextExcluding(bare_markers_v1([tool_open, function_open])),
                Expr::seq([
                    Expr::TextUntil {
                        end: tool_open.as_bytes().to_vec(),
                        avoid: bare_markers_v1([function_open]),
                    },
                    Expr::literal(after_open),
                    calls,
                ]),
            ])
        }
    };
    let root = builder.rule(body);
    builder.finish(root).map_err(|error| grammar_error(&error))
}

fn call_after_open(
    builder: &mut GrammarBuilder,
    descriptor: &FormatDescriptorV1,
    format: &FrontendParserDeclarationV1,
    tool: &FrontendToolV1,
) -> Result<Expr, FrontendErrorV1> {
    let schema = tool.parameters.to_json_value();
    let charset = &descriptor.tools.name_bytes;
    let name_bytes: Vec<u8> = (0..=127_u8).filter(|byte| charset.permits(*byte)).collect();
    let maximum_name = descriptor.limits.maximum_name_bytes as u32;
    let free_name = || name_class_v1(&name_bytes).repeat(1, Some(maximum_name));
    // A value is followed by the close marker. Free text ends at the first
    // bare close marker, as the decoder reads it, including one that begins
    // inside the text, and cannot hold a bare marker that would end it
    // sooner.
    // The typed grammar closes a JSON value's name with its own close.
    let parameter = |name: Expr, (value_and_close, json): (Expr, bool)| {
        let name_close =
            if json && format.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
                &format.parameter_json_name_close
            } else {
                &format.parameter_name_close
            };
        Expr::seq([
            Expr::literal(&format.parameter_open),
            name,
            Expr::literal(name_close),
            Expr::literal(&format.parameter_value_open),
            value_and_close,
        ])
    };
    let close = || Expr::literal(&format.parameter_close);
    let (bare_close, after_close) = bare_v1(&format.parameter_close);
    // A function's arguments end at its close; a key-value call's arguments
    // end at the call close, and a value may hold the next key's open marker.
    // A typed grammar's raw value runs to its own close whatever it holds.
    let avoid = match format.tool_grammar {
        FrontendToolGrammarV1::XmlTypedParametersV1 => Vec::new(),
        FrontendToolGrammarV1::XmlFunctionParametersV1 => {
            bare_markers_v1([format.parameter_open.trim(), format.function_close.trim()])
        }
        FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => {
            bare_markers_v1([format.tool_call_close.trim()])
        }
    };
    let free_text = || {
        Expr::seq([
            Expr::TextUntil {
                end: bare_close.as_bytes().to_vec(),
                avoid: avoid.clone(),
            },
            Expr::literal(after_close),
        ])
    };
    // A literal value the grammar offers must read back whole under the
    // parameter's name, as the served parser reads the call it is in.
    let served = std::sync::Arc::new(descriptor.clone());
    let literal = |name: &str| {
        let name = name.to_owned();
        let served = std::sync::Arc::clone(&served);
        move |value: &str| literal_value_reads_back_v1(&served, tool, &name, value)
    };
    let parameters = if tool.strict == Some(false) || !tool.parameters_present {
        // Arguments are not constrained: any parameters, as free text.
        Expr::OrderedSubset {
            items: Vec::new(),
            tail: Some(Box::new(parameter(free_name(), (free_text(), false)))),
            separator: Box::new(Expr::Empty),
        }
    } else {
        let Value::Object(fields) = &schema else {
            return Err(FrontendErrorV1::new(
                "tool parameters must be a JSON schema object",
                Some("tools"),
            ));
        };
        let required: Vec<&str> = fields
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let properties = fields.get("properties").and_then(Value::as_object);
        let mut compiler = SchemaCompiler::new(builder, &schema, JsonOptions::default());
        // The typed grammar reads a fraction or exponent beyond a finite
        // binary64 back as text, so a constrained number never spells one.
        if format.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
            compiler = compiler.finite_numbers();
        }
        let mut items = Vec::new();
        // The names written once each: the declared properties, then the
        // required names that are not properties.
        let mut written: Vec<&str> = Vec::new();
        // The parser reads only a parameter name in the declared byte set
        // and length: a required name outside it cannot be written so that
        // it reads back, and an optional one is never written.
        let readable = |name: &str| validate_name_v1(descriptor, name, "parameter").is_ok();
        if let Some(name) = required.iter().find(|name| !readable(name)) {
            return Err(FrontendErrorV1::new(
                format!(
                    "tools: the required parameter {name:?} of {} is not a name the tool-call \
                     parser reads",
                    tool.name
                ),
                Some("tools"),
            ));
        }
        for (name, property) in properties.into_iter().flatten() {
            written.push(name);
            if !readable(name) {
                continue;
            }
            let value = parameter_value(
                &mut compiler,
                property,
                &free_text,
                &close,
                &literal(name),
                name,
            )?;
            items.push((
                parameter(Expr::literal(name), value),
                required.contains(&name.as_str()),
            ));
        }
        // A required name that is not a property, and any other name, has
        // the schema the decoder resolves it by.
        for name in &required {
            if !written.contains(name) {
                written.push(name);
                let value = match additional_schema_v1(fields) {
                    ParameterSchemaV1::Typed(schema) => parameter_value(
                        &mut compiler,
                        schema,
                        &free_text,
                        &close,
                        &literal(name),
                        name,
                    )?,
                    ParameterSchemaV1::Free => (free_text(), false),
                    ParameterSchemaV1::Refused => (Expr::Nothing, false),
                };
                items.push((parameter(Expr::literal(name), value), true));
            }
        }
        // Declared properties with no additional-property schema admit no
        // other name. Any other name is none of the names written above: the
        // decoder reads a declared name by its property's schema, not the
        // tail's, and a repeated name replaces the earlier value.
        let other_name = || names_except_v1(&name_bytes, &written, 1, maximum_name);
        let closed = properties.is_some()
            && !fields.contains_key("patternProperties")
            && !fields.contains_key("additionalProperties");
        let tail = match additional_schema_v1(fields) {
            _ if closed => None,
            ParameterSchemaV1::Typed(schema) => {
                // Any other name reads its value the same way; one that is
                // not a declared property stands for them.
                let mut other = String::from("x");
                while properties.is_some_and(|properties| properties.contains_key(&other)) {
                    other.push('x');
                }
                Some(parameter(
                    other_name(),
                    parameter_value(
                        &mut compiler,
                        schema,
                        &free_text,
                        &close,
                        &literal(&other),
                        "an additional property",
                    )?,
                ))
            }
            ParameterSchemaV1::Free => Some(parameter(other_name(), (free_text(), false))),
            ParameterSchemaV1::Refused => None,
        };
        Expr::OrderedSubset {
            items,
            tail: tail.map(Box::new),
            separator: Box::new(Expr::Empty),
        }
    };
    // The call open marker is the caller's, so free text can end with it.
    // The typed grammar's caller closes the block after its last call.
    let block_close = if format.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
        Expr::Empty
    } else {
        Expr::literal(&format.tool_call_close)
    };
    Ok(Expr::seq([
        Expr::literal(&format.function_open),
        Expr::literal(&tool.name),
        Expr::literal(&format.function_name_close),
        parameters,
        Expr::literal(&format.function_close),
        block_close,
    ]))
}

/// A parameter's value text and its close marker, and whether the value is
/// JSON. Anything a string could be is raw text, as the decoder keeps it;
/// string enumerations are their raw values; every other schema is JSON.
///
/// A string enumeration or constant value is written as it is, so each must
/// read back whole (`reads_back`): one that spells a marker ending a raw
/// value, or opens a string literal a JSON value's scan never closes, would
/// parse as another argument. Such a schema cannot be constrained, and is
/// refused, rather than letting a forced call change its argument.
fn parameter_value(
    compiler: &mut SchemaCompiler<'_>,
    schema: &Value,
    free_text: &dyn Fn() -> Expr,
    close: &dyn Fn() -> Expr,
    reads_back: &dyn Fn(&str) -> bool,
    name: &str,
) -> Result<(Expr, bool), FrontendErrorV1> {
    let Value::Object(fields) = schema else {
        return Ok((free_text(), false));
    };
    let values = fields
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| fields.get("const").map(|value| vec![value.clone()]));
    if let Some(values) = values
        && values.iter().all(Value::is_string)
    {
        if let Some(value) = values
            .iter()
            .filter_map(Value::as_str)
            .find(|value| !reads_back(value))
        {
            return Err(FrontendErrorV1::new(
                format!(
                    "tools: the value {value:?} of {name} cannot be written so that the tool-call \
                     parser reads it back"
                ),
                Some("tools"),
            ));
        }
        let values = Expr::alt(
            values
                .iter()
                .map(|value| Expr::literal(value.as_str().expect("checked string"))),
        );
        return Ok((Expr::seq([values, close()]), false));
    }
    if admits_string_v1(schema) {
        return Ok((free_text(), false));
    }
    let value = compiler
        .schema(schema)
        .map_err(|error| grammar_error(&error))?;
    Ok((Expr::seq([value, close()]), true))
}

/// One byte of `bytes`.
fn name_class_v1(bytes: &[u8]) -> Expr {
    Expr::Class(CharClass::from_ranges(
        bytes
            .iter()
            .map(|byte| (u32::from(*byte), u32::from(*byte))),
    ))
}

/// The names of `minimum` to `maximum` bytes of `bytes` other than those in
/// `taken`. With nothing taken that such a name could spell, any such name,
/// as one bounded run of the byte class.
fn names_except_v1(bytes: &[u8], taken: &[&str], minimum: u32, maximum: u32) -> Expr {
    let taken: BTreeSet<&[u8]> = taken
        .iter()
        .map(|name| name.as_bytes())
        .filter(|name| name.iter().all(|byte| bytes.contains(byte)))
        .collect();
    words_except_v1(bytes, &taken, minimum as usize, maximum as usize)
}

/// The words of `minimum` to `maximum` bytes of `bytes` that are not in
/// `taken`.
///
/// A trie of the taken words would nest two expressions for each byte of
/// the longest word, past the depth the grammar wire admits for the longest
/// names a declaration permits. Instead the words split at `half` the bound:
/// the words shorter than it, and a first half followed by a rest. A first
/// half that begins no taken word is followed by any rest; a taken first
/// half is followed by the rests its taken words do not end in, written once
/// for every first half that shares them. Each level halves the bound, so
/// the expression grows as the bound times its logarithm for each distinct
/// taken prefix, and nests as deep as that logarithm.
fn words_except_v1(bytes: &[u8], taken: &BTreeSet<&[u8]>, minimum: usize, maximum: usize) -> Expr {
    let taken: BTreeSet<&[u8]> = taken
        .iter()
        .copied()
        .filter(|word| (minimum..=maximum).contains(&word.len()))
        .collect();
    if minimum > maximum {
        return Expr::Nothing;
    }
    if taken.is_empty() {
        return name_class_v1(bytes).repeat(minimum as u32, Some(maximum as u32));
    }
    if maximum <= 1 {
        let mut words = Vec::new();
        if minimum == 0 && !taken.contains(&b""[..]) {
            words.push(Expr::Empty);
        }
        let other: Vec<u8> = bytes
            .iter()
            .copied()
            .filter(|byte| maximum == 1 && !taken.contains(&[*byte][..]))
            .collect();
        if !other.is_empty() {
            words.push(name_class_v1(&other));
        }
        return Expr::alt(words);
    }
    let half = maximum.div_ceil(2);
    let mut words = vec![words_except_v1(bytes, &taken, minimum, half - 1)];
    let (rest_minimum, rest_maximum) = (minimum.saturating_sub(half), maximum - half);
    let mut rests: BTreeMap<&[u8], BTreeSet<&[u8]>> = BTreeMap::new();
    for word in taken.iter().filter(|word| word.len() >= half) {
        let (first, rest) = word.split_at(half);
        rests.entry(first).or_default().insert(rest);
    }
    if rests.is_empty() {
        words.push(name_class_v1(bytes).repeat(half as u32, Some(maximum as u32)));
    } else {
        let firsts: BTreeSet<&[u8]> = rests.keys().copied().collect();
        words.push(Expr::seq([
            words_except_v1(bytes, &firsts, half, half),
            name_class_v1(bytes).repeat(rest_minimum as u32, Some(rest_maximum as u32)),
        ]));
    }
    let mut shared: BTreeMap<BTreeSet<&[u8]>, Vec<&[u8]>> = BTreeMap::new();
    for (first, rest) in rests {
        shared.entry(rest).or_default().push(first);
    }
    for (rest, firsts) in shared {
        words.push(Expr::seq([
            Expr::alt(
                firsts
                    .into_iter()
                    .map(|first| Expr::Literal(first.to_vec())),
            ),
            words_except_v1(bytes, &rest, rest_minimum, rest_maximum),
        ]));
    }
    Expr::alt(words)
}

/// A declared marker as the bare marker the decoder reads, and the
/// whitespace the template writes after it.
fn bare_v1(marker: &str) -> (&str, &str) {
    let bare = marker.trim();
    let end = marker.len() - marker.trim_start().len() + bare.len();
    (bare, &marker[end..])
}

/// The non-empty bare markers among `markers`, as bytes.
fn bare_markers_v1<'a>(markers: impl IntoIterator<Item = &'a str>) -> Vec<Vec<u8>> {
    markers
        .into_iter()
        .filter(|marker| !marker.is_empty())
        .map(|marker| marker.as_bytes().to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use runtime_grammar::{
        CompiledGrammar, Grammar, GrammarMatcher, TokenVocabulary, VocabularyToken,
    };

    use super::names_except_v1;

    /// Every word of up to one byte past the bound, over the name bytes and
    /// one byte outside them, under taken sets of short and bound-length
    /// names, shared prefixes, names the bytes cannot spell and the empty
    /// name: the names admitted are exactly the names of one to the bound
    /// bytes that are not taken.
    #[test]
    fn names_except_admits_exactly_the_other_names() {
        let bytes = b"_ab";
        let alphabet = b"_abc";
        let tokens = (0..=255_u8)
            .map(|byte| VocabularyToken {
                bytes: vec![byte],
                special: false,
            })
            .collect();
        let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).unwrap());
        // The words over the name bytes, and those with the outside byte
        // up to three bytes long.
        let mut words = vec![Vec::new()];
        for length in 1..=7 {
            let shorter: Vec<Vec<u8>> = words
                .iter()
                .filter(|word| word.len() == length - 1)
                .cloned()
                .collect();
            for word in shorter {
                for byte in alphabet {
                    if *byte != b'c' || length <= 3 {
                        words.push([word.as_slice(), &[*byte]].concat());
                    }
                }
            }
        }
        let mut sets: Vec<Vec<String>> = [
            &[][..],
            &["a"],
            &["", "_"],
            &["ab", "aba", "abab", "ababab"],
            &["aaaaaa", "aaaaab", "aaaab_", "b"],
            &["ac", "a", "abcab"],
            &["__", "_a", "_b", "a_", "aa", "ab", "b_", "ba", "bb"],
        ]
        .iter()
        .map(|set| set.iter().map(|name| (*name).to_owned()).collect())
        .collect();
        // Pseudo-random sets, fixed by the seed.
        let mut seed = 0x2545_f491_u32;
        let mut next = |bound: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed % bound
        };
        for _ in 0..40 {
            let count = 1 + next(6);
            sets.push(
                (0..count)
                    .map(|_| {
                        let length = next(8);
                        (0..length)
                            .map(|_| char::from(alphabet[next(3) as usize]))
                            .collect()
                    })
                    .collect(),
            );
        }
        for maximum in [1, 2, 3, 5, 6] {
            for set in &sets {
                let taken: Vec<&str> = set.iter().map(String::as_str).collect();
                let grammar = Grammar::from_expr(names_except_v1(bytes, &taken, 1, maximum));
                let compiled =
                    Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
                for word in words
                    .iter()
                    .filter(|word| word.len() <= maximum as usize + 1)
                {
                    let mut matcher = GrammarMatcher::new(Arc::clone(&compiled));
                    let admitted = matcher.accept_bytes(word).is_ok() && matcher.is_complete();
                    let expected = (1..=maximum as usize).contains(&word.len())
                        && word.iter().all(|byte| bytes.contains(byte))
                        && !taken.contains(&std::str::from_utf8(word).unwrap());
                    assert_eq!(
                        admitted,
                        expected,
                        "{taken:?} up to {maximum}: {:?}",
                        String::from_utf8_lossy(word)
                    );
                }
            }
        }
    }
}
