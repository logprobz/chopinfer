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
//! and an empty marker is written as nothing and avoided nowhere.

use runtime_grammar::{
    CharClass, Expr, Grammar, GrammarBuilder, GrammarError, JsonOptions, SchemaCompiler,
};
use serde_json::Value;

use crate::chatml_decode::{
    ParameterSchemaV1, additional_schema_v1, admits_string_v1, literal_value_reads_back_v1,
};
use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::request::{FrontendErrorV1, FrontendToolChoiceV1, FrontendToolV1};

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
    let mut builder = GrammarBuilder::new();
    let mut rests = Vec::with_capacity(offered.len());
    for tool in offered {
        rests.push(call_after_open(&mut builder, descriptor, format, tool)?);
    }
    // Everything a call writes after its open marker.
    let rest = Expr::alt(rests);
    let open = || Expr::literal(&format.tool_call_open);
    let call = Expr::seq([open(), rest.clone()]);
    let more_calls = Expr::seq([Expr::literal(&format.tool_call_separator), call.clone()]).star();
    // A template that writes a prefix before the answer may start with it.
    let answer_prefix = Expr::literal(&format.answer_prefix).optional();
    let body = match forced {
        FrontendToolChoiceV1::Named(_) => Expr::seq([answer_prefix, call]),
        FrontendToolChoiceV1::Required => Expr::seq([answer_prefix, call, more_calls]),
        // Free text, then optionally calls. The decoder reads everything from
        // the first call or function marker on as calls, so the text contains
        // neither and, when calls follow, ends at the first call marker.
        _ => {
            let (tool_open, after_open) = bare_v1(&format.tool_call_open);
            let function_open = format.function_open.trim();
            Expr::alt([
                Expr::TextExcluding(bare_markers_v1([tool_open, function_open])),
                Expr::seq([
                    Expr::TextUntil {
                        end: tool_open.as_bytes().to_vec(),
                        avoid: bare_markers_v1([function_open]),
                    },
                    Expr::literal(after_open),
                    rest,
                    more_calls,
                ]),
            ])
        }
    };
    let root = builder.rule(body);
    let grammar = builder
        .finish(root)
        .map_err(|error| grammar_error(&error))?;
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

fn call_after_open(
    builder: &mut GrammarBuilder,
    descriptor: &FormatDescriptorV1,
    format: &FrontendParserDeclarationV1,
    tool: &FrontendToolV1,
) -> Result<Expr, FrontendErrorV1> {
    let schema = tool.parameters.to_json_value();
    let free_name = || {
        let charset = &descriptor.tools.name_bytes;
        let bytes = (0..=127_u8)
            .filter(|byte| charset.permits(*byte))
            .map(|byte| (u32::from(byte), u32::from(byte)));
        Expr::Class(CharClass::from_ranges(bytes))
            .repeat(1, Some(descriptor.limits.maximum_name_bytes as u32))
    };
    // A value is followed by the close marker. Free text ends at the first
    // bare close marker, as the decoder reads it, including one that begins
    // inside the text, and cannot hold a bare marker that would end it
    // sooner.
    let parameter = |name: Expr, value_and_close: Expr| {
        Expr::seq([
            Expr::literal(&format.parameter_open),
            name,
            Expr::literal(&format.parameter_name_close),
            Expr::literal(&format.parameter_value_open),
            value_and_close,
        ])
    };
    let close = || Expr::literal(&format.parameter_close);
    let (bare_close, after_close) = bare_v1(&format.parameter_close);
    // A function's arguments end at its close; a key-value call's arguments
    // end at the call close, and a value may hold the next key's open marker.
    let avoid = match format.tool_grammar {
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
            tail: Some(Box::new(parameter(free_name(), free_text()))),
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
        let mut items = Vec::new();
        for (name, property) in properties.into_iter().flatten() {
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
            if !properties.is_some_and(|properties| properties.contains_key(*name)) {
                let value = match additional_schema_v1(fields) {
                    ParameterSchemaV1::Typed(schema) => parameter_value(
                        &mut compiler,
                        schema,
                        &free_text,
                        &close,
                        &literal(name),
                        name,
                    )?,
                    ParameterSchemaV1::Free => free_text(),
                    ParameterSchemaV1::Refused => Expr::Nothing,
                };
                items.push((parameter(Expr::literal(name), value), true));
            }
        }
        // Declared properties with no additional-property schema admit no
        // other name.
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
                    free_name(),
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
            ParameterSchemaV1::Free => Some(parameter(free_name(), free_text())),
            ParameterSchemaV1::Refused => None,
        };
        Expr::OrderedSubset {
            items,
            tail: tail.map(Box::new),
            separator: Box::new(Expr::Empty),
        }
    };
    // The call open marker is the caller's, so free text can end with it.
    Ok(Expr::seq([
        Expr::literal(&format.function_open),
        Expr::literal(&tool.name),
        Expr::literal(&format.function_name_close),
        parameters,
        Expr::literal(&format.function_close),
        Expr::literal(&format.tool_call_close),
    ]))
}

/// A parameter's value text and its close marker. Anything a string could be
/// is raw text, as the decoder keeps it; string enumerations are their raw
/// values; every other schema is JSON.
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
) -> Result<Expr, FrontendErrorV1> {
    let Value::Object(fields) = schema else {
        return Ok(free_text());
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
        return Ok(Expr::seq([values, close()]));
    }
    if admits_string_v1(schema) {
        return Ok(free_text());
    }
    let value = compiler
        .schema(schema)
        .map_err(|error| grammar_error(&error))?;
    Ok(Expr::seq([value, close()]))
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
