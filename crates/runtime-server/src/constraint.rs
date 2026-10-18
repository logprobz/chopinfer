//! Structured-output fields, read into the grammar a response must match.
//!
//! `structured_outputs` names one constraint and `response_format` overrides
//! its JSON fields, as the reference API combines them. The grammar is built
//! here, before admission, so an unsupported constraint is a 400 that names
//! the field it came from.

use std::sync::Arc;

use runtime_grammar::{
    Grammar, GrammarError, JsonOptions, choice_grammar, json_object_grammar, json_schema_grammar,
    regex_grammar,
};
use serde_json::Value;

use crate::error::ApiError;
use crate::frontend::FrontendJsonV1;

const STRUCTURED_OUTPUTS: &str = "structured_outputs";
const RESPONSE_FORMAT: &str = "response_format";
const FIELDS: &[&str] = &[
    "json",
    "regex",
    "choice",
    "grammar",
    "json_object",
    "structural_tag",
    "disable_any_whitespace",
    "disable_additional_properties",
    "whitespace_pattern",
];

/// The one constraint a request selected, with the field it came from.
enum Kind {
    Json(Value),
    JsonObject,
    Regex(String),
    Choice(Vec<String>),
}

/// What a request's structured-output fields ask for.
pub(crate) struct RequestedConstraint {
    kind: Kind,
    source: &'static str,
    options: JsonOptions,
    /// Whether `structured_outputs` itself names `json`, `regex` or
    /// `choice`, which the reference refuses together with a named tool.
    pub(crate) excludes_named_tools: bool,
}

impl RequestedConstraint {
    pub(crate) fn grammar(&self) -> Result<Arc<Grammar>, ApiError> {
        let grammar = match &self.kind {
            Kind::Json(schema) => json_schema_grammar(schema, self.options),
            Kind::JsonObject => Ok(json_object_grammar(self.options)),
            Kind::Regex(pattern) => regex_grammar(pattern),
            Kind::Choice(choices) => choice_grammar(choices),
        };
        grammar
            .map(Arc::new)
            .map_err(|error| grammar_error(&error, self.source))
    }
}

pub(crate) fn grammar_error(error: &GrammarError, parameter: &str) -> ApiError {
    ApiError::invalid_owned(format!("{parameter}: {error}"), Some(parameter.to_owned()))
}

fn invalid(message: &str, parameter: &'static str) -> ApiError {
    ApiError::invalid(message.to_owned(), Some(parameter))
}

fn entry<'a>(entries: &'a [(String, FrontendJsonV1)], name: &str) -> Option<&'a FrontendJsonV1> {
    entries
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .filter(|value| !matches!(value, FrontendJsonV1::Null))
}

/// Reads both fields. `None` means the output is unconstrained.
///
/// With `strict_only` ([`crate::ServerFrontendV1::constrains_strict_only`])
/// a `json_object` response format, or a JSON schema one without `strict:
/// true`, is checked and constrains nothing: it neither selects nor
/// replaces a constraint, so an explicit `structured_outputs` one stands.
pub(crate) fn requested_constraint(
    structured_outputs: Option<&FrontendJsonV1>,
    response_format: Option<&FrontendJsonV1>,
    strict_only: bool,
) -> Result<Option<RequestedConstraint>, ApiError> {
    let structured = match structured_outputs {
        None | Some(FrontendJsonV1::Null) => None,
        Some(FrontendJsonV1::Object(entries)) => Some(entries.as_slice()),
        Some(_) => {
            return Err(invalid(
                "structured_outputs must be an object",
                STRUCTURED_OUTPUTS,
            ));
        }
    };
    let mut kinds: Vec<(&str, &'static str)> = Vec::new();
    let mut options = JsonOptions::default();
    let mut json = None;
    let mut regex = None;
    let mut choice = None;
    let mut excludes_named_tools = false;
    let mut expects_constraint = false;
    if let Some(entries) = structured {
        if let Some((name, _)) = entries
            .iter()
            .find(|(name, _)| !FIELDS.contains(&name.as_str()))
        {
            return Err(ApiError::invalid_owned(
                format!("structured_outputs.{name} is not a recognized field"),
                Some(STRUCTURED_OUTPUTS.to_owned()),
            ));
        }
        for name in ["grammar", "structural_tag"] {
            if entry(entries, name).is_some() {
                return Err(ApiError::invalid_owned(
                    format!("structured_outputs.{name} is not supported"),
                    Some(STRUCTURED_OUTPUTS.to_owned()),
                ));
            }
        }
        if let Some(value) = entry(entries, "disable_any_whitespace") {
            let FrontendJsonV1::Bool(disable) = value else {
                return Err(invalid(
                    "structured_outputs.disable_any_whitespace must be a boolean",
                    STRUCTURED_OUTPUTS,
                ));
            };
            options.any_whitespace = !disable;
            options.max_whitespace_cnt = None;
        }
        // A declared `properties` without `additionalProperties` already
        // admits no other keys, which is what this option asks for, so either
        // value compiles the same grammar.
        if let Some(value) = entry(entries, "disable_additional_properties")
            && !matches!(value, FrontendJsonV1::Bool(_))
        {
            return Err(invalid(
                "structured_outputs.disable_additional_properties must be a boolean",
                STRUCTURED_OUTPUTS,
            ));
        }
        // vLLM accepts a pattern and no backend reads it, so it is ignored
        // here too, and said so once.
        match entry(entries, "whitespace_pattern") {
            None => {}
            Some(FrontendJsonV1::String(_)) => {
                static IGNORED: std::sync::Once = std::sync::Once::new();
                IGNORED.call_once(|| {
                    eprintln!(
                        "debug: structured_outputs.whitespace_pattern is accepted and ignored, \
                         as vLLM does"
                    );
                });
            }
            Some(_) => {
                return Err(invalid(
                    "structured_outputs.whitespace_pattern must be a string",
                    STRUCTURED_OUTPUTS,
                ));
            }
        }
        if let Some(value) = entry(entries, "json") {
            json = Some(match value {
                FrontendJsonV1::String(text) => {
                    serde_json::from_str::<Value>(text).map_err(|_| {
                        invalid(
                            "structured_outputs.json is not valid JSON",
                            STRUCTURED_OUTPUTS,
                        )
                    })?
                }
                FrontendJsonV1::Object(_) | FrontendJsonV1::Bool(_) => value.to_json_value(),
                _ => {
                    return Err(invalid(
                        "structured_outputs.json must be a schema object or its JSON text",
                        STRUCTURED_OUTPUTS,
                    ));
                }
            });
            kinds.push(("json", STRUCTURED_OUTPUTS));
        }
        if let Some(value) = entry(entries, "regex") {
            let FrontendJsonV1::String(pattern) = value else {
                return Err(invalid(
                    "structured_outputs.regex must be a string",
                    STRUCTURED_OUTPUTS,
                ));
            };
            regex = Some(pattern.clone());
            kinds.push(("regex", STRUCTURED_OUTPUTS));
        }
        if let Some(value) = entry(entries, "choice") {
            let FrontendJsonV1::Array(values) = value else {
                return Err(invalid(
                    "structured_outputs.choice must be a list of strings",
                    STRUCTURED_OUTPUTS,
                ));
            };
            let values = values
                .iter()
                .map(|value| match value {
                    FrontendJsonV1::String(text) => Ok(text.clone()),
                    _ => Err(invalid(
                        "structured_outputs.choice must be a list of strings",
                        STRUCTURED_OUTPUTS,
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            choice = Some(values);
            kinds.push(("choice", STRUCTURED_OUTPUTS));
        }
        excludes_named_tools = !kinds.is_empty();
        if let Some(value) = entry(entries, "json_object") {
            if !matches!(value, FrontendJsonV1::Bool(true)) {
                return Err(invalid(
                    "structured_outputs.json_object must be true when given",
                    STRUCTURED_OUTPUTS,
                ));
            }
            kinds.push(("json_object", STRUCTURED_OUTPUTS));
        }
    }
    // response_format replaces the matching structured_outputs field.
    match response_format {
        None | Some(FrontendJsonV1::Null) => {}
        Some(FrontendJsonV1::Object(entries)) => {
            let kind = match entry(entries, "type") {
                Some(FrontendJsonV1::String(kind)) => kind.as_str(),
                _ => {
                    return Err(invalid(
                        "response_format.type must be a string",
                        RESPONSE_FORMAT,
                    ));
                }
            };
            match kind {
                "text" => {}
                "json_object" if strict_only => {}
                "json_object" => {
                    kinds.retain(|(name, _)| *name != "json_object");
                    kinds.push(("json_object", RESPONSE_FORMAT));
                }
                "json_schema" => {
                    let Some(FrontendJsonV1::Object(format)) = entry(entries, "json_schema") else {
                        return Err(invalid(
                            "When response_format type is 'json_schema', the 'json_schema' field \
                             must be provided.",
                            RESPONSE_FORMAT,
                        ));
                    };
                    if !matches!(entry(format, "name"), Some(FrontendJsonV1::String(_))) {
                        return Err(invalid(
                            "response_format.json_schema.name must be a string",
                            RESPONSE_FORMAT,
                        ));
                    }
                    let strict = match entry(format, "strict") {
                        None => false,
                        Some(FrontendJsonV1::Bool(strict)) => *strict,
                        Some(_) => {
                            return Err(invalid(
                                "response_format.json_schema.strict must be a boolean",
                                RESPONSE_FORMAT,
                            ));
                        }
                    };
                    if strict || !strict_only {
                        kinds.retain(|(name, _)| *name != "json");
                        match entry(format, "schema") {
                            Some(schema) => {
                                json = Some(schema.to_json_value());
                                kinds.push(("json", RESPONSE_FORMAT));
                            }
                            // The reference replaces json with nothing, which
                            // leaves a constraint object that names none.
                            None => {
                                json = None;
                                expects_constraint = true;
                            }
                        }
                    }
                }
                "structural_tag" => {
                    return Err(invalid(
                        "response_format structural_tag is not supported",
                        RESPONSE_FORMAT,
                    ));
                }
                _ => {
                    return Err(ApiError::invalid_owned(
                        format!("response_format type '{kind}' is not supported"),
                        Some(RESPONSE_FORMAT.to_owned()),
                    ));
                }
            }
        }
        Some(_) => {
            return Err(invalid(
                "response_format must be an object",
                RESPONSE_FORMAT,
            ));
        }
    }
    if kinds.len() > 1 {
        return Err(invalid(
            "You can only use one kind of structured outputs constraint but multiple are \
             specified",
            kinds.last().expect("two constraints").1,
        ));
    }
    let Some(&(name, source)) = kinds.first() else {
        if structured.is_some() || expects_constraint {
            return Err(invalid(
                "You must use one kind of structured outputs constraint but none are specified",
                STRUCTURED_OUTPUTS,
            ));
        }
        return Ok(None);
    };
    let kind = match name {
        "json" => Kind::Json(json.expect("json was recorded")),
        "json_object" => Kind::JsonObject,
        "regex" => Kind::Regex(regex.expect("regex was recorded")),
        _ => Kind::Choice(choice.expect("choice was recorded")),
    };
    Ok(Some(RequestedConstraint {
        kind,
        source,
        options,
        excludes_named_tools,
    }))
}
