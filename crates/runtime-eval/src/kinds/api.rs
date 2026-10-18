//! `api.v1`: OpenAI-compatible API behavior, one row per case: completions,
//! chat, streaming, usage, token IDs, logprobs, tool calls and metrics
//! (docs/check-kinds.md#api-apiv1). The capture is `tools/qual/api.py`:
//! the same requests in the same order, each case failing where and with
//! the message Python's would.

use serde_json::{Map, Value, json};

use crate::capture::{self, receipt, thinking_control};
use crate::declaration::Check;
use crate::http::{Client, HttpError};
use crate::json::{self, python_eq, truthy};
use crate::kind::{Capture, Kind, KindError, KindInfo, SHARED_KINDS, default_score};
use crate::kinds::{applied_constants, constant, get, invalid, require};
use crate::repr::{self, type_name};

pub struct Api;

/// The api.v1 cases in the order `api.py` runs them (`kinds.API_CASES`).
pub const API_CASES: [&str; 40] = [
    "choice_single_False",
    "multiple_choices_False_False",
    "multiple_choices_False_True",
    "choice_single_True",
    "multiple_choices_True_False",
    "multiple_choices_True_True",
    "generated_token_ids_False_False",
    "generated_token_ids_False_True",
    "generated_token_ids_True_False",
    "generated_token_ids_True_True",
    "tokenize",
    "batch_single",
    "batch_False_False",
    "batch_False_True",
    "batch_single",
    "batch_True_False",
    "batch_True_True",
    "add_special_tokens_False_False",
    "add_special_tokens_True_False",
    "add_special_tokens_False_True",
    "add_special_tokens_True_True",
    "detokenize",
    "stop_token_ids",
    "generation_prompt_control",
    "hidden_reasoning",
    "ignore_eos",
    "nullable_flags",
    "unknown_extension",
    "sampling_extensions",
    "chat_token_limit_precedence",
    "continuous_usage",
    "stream_usage",
    "stop",
    "completion_logprobs",
    "reasoning_True",
    "reasoning_False",
    "stream_tool_arguments",
    "tool_call",
    "tool_result",
    "metrics_compatibility",
];

impl Kind for Api {
    fn info(&self) -> &'static KindInfo {
        &SHARED_KINDS[4]
    }

    /// `api.main`: the 40 cases in order against the server, then the
    /// receipt. A case the server answers with an error status or a
    /// response the case rejects fails its row; an unreachable server or a
    /// spent budget is the environment's.
    fn capture(&self, capture: &mut Capture<'_>) -> Result<Value, KindError> {
        applied_constants(capture.check, &[("api", "TOOL_RESULT_MAX_TOKENS")])?;
        let tool_result_max_tokens = constant(
            capture.check,
            "api",
            "TOOL_RESULT_MAX_TOKENS",
            json!(TOOL_RESULT_MAX_TOKENS),
        );
        let (path, conversations) = capture::input_json(capture, "chat_renders")?;
        let tool_case = conversations
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["id"] == "tools-00"))
            .cloned()
            .ok_or_else(|| {
                KindError::Insufficient(format!("{}: no tools-00 conversation", path.display()))
            })?;
        let model = json!(capture.server.model);
        let mechanics = capture.mechanics;
        let mut run = Run {
            capture,
            rows: Vec::new(),
            errors: Vec::new(),
        };
        // Python's errors outside a case stop the run before its receipt.
        let crashed = |error: Failure| invalid(error);
        let hello = json!([{"role": "user", "content": "Hello"}]);
        let with_input = |mut body: Value, chat: bool| {
            if chat {
                body["messages"] = hello.clone();
            } else {
                body["prompt"] = json!("Hello");
            }
            body
        };
        let streamed = |mut body: Value, stream: bool| {
            body["stream"] = json!(stream);
            if stream {
                body["stream_options"] = json!({"include_usage": true});
            }
            body
        };
        for chat in [false, true] {
            // vLLM refuses n > 1 with greedy sampling, so multiple choices sample.
            let body = with_input(
                json!({"model": model, "n": 3, "seed": 41, "temperature": 0.7, "max_tokens": 3,
                       "ignore_eos": true}),
                chat,
            );
            let mut single_body = body.clone();
            single_body["n"] = json!(1);
            let single = run.check(
                &format!("choice_single_{}", py::name(chat)),
                single_body,
                None,
                |value| {
                    need(
                        py::len(&py::item(value, "choices")?)? == 1,
                        "missing choice",
                    )
                },
                None,
            )?;
            let expected_prompt = match single.filter(truthy) {
                Some(single) => py::item(
                    &py::item(&single, "usage").map_err(crashed)?,
                    "prompt_tokens",
                )
                .map_err(crashed)?,
                None => Value::Null,
            };
            for stream in [false, true] {
                run.check(
                    &format!("multiple_choices_{}_{}", py::name(chat), py::name(stream)),
                    streamed(body.clone(), stream),
                    None,
                    |value| validate_choices(value, stream, 3, &expected_prompt, 9),
                    None,
                )?;
            }
        }
        for chat in [false, true] {
            for stream in [false, true] {
                let mut request = with_input(
                    json!({"model": model, "return_token_ids": true, "temperature": 0,
                           "max_tokens": 3, "ignore_eos": true, "stream": stream}),
                    chat,
                );
                if stream {
                    request["stream_options"] = json!({"include_usage": true});
                }
                run.check(
                    &format!(
                        "generated_token_ids_{}_{}",
                        py::name(chat),
                        py::name(stream)
                    ),
                    request,
                    None,
                    |value| validate_token_ids(value, stream, chat),
                    None,
                )?;
            }
        }
        let encoded = run.check(
            "tokenize",
            json!({"model": model, "prompt": "Hello", "add_special_tokens": false}),
            Some("/tokenize"),
            |value| {
                let count = py::item(value, "count")?;
                let length = py::len(&py::item(value, "tokens")?)?;
                need(python_eq(&count, &json!(length)), "token count differs")
            },
            None,
        )?;
        let tokens = match &encoded {
            Some(encoded) => Some(py::item(encoded, "tokens").map_err(crashed)?),
            None => None,
        };
        if let Some(tokens) = tokens.as_ref().filter(|tokens| truthy(tokens)) {
            for prompt in [json!(["Hello", "Hello"]), json!([tokens, tokens])] {
                let single = run.check(
                    "batch_single",
                    json!({"model": model, "prompt": prompt[0], "max_tokens": 1, "temperature": 0}),
                    None,
                    |value| need(truthy(&py::item(value, "choices")?), "missing choice"),
                    None,
                )?;
                let expected_prompt = match single.filter(truthy) {
                    Some(single) => py::double(
                        &py::item(
                            &py::item(&single, "usage").map_err(crashed)?,
                            "prompt_tokens",
                        )
                        .map_err(crashed)?,
                    )
                    .map_err(crashed)?,
                    None => Value::Null,
                };
                for stream in [false, true] {
                    let mut request = json!({"model": model, "prompt": prompt, "n": 2, "seed": 41,
                                             "temperature": 0.7, "max_tokens": 3,
                                             "ignore_eos": true, "stream": stream});
                    if stream {
                        request["stream_options"] = json!({"include_usage": true});
                    }
                    run.check(
                        &format!(
                            "batch_{}_{}",
                            py::name(prompt[0].is_array()),
                            py::name(stream)
                        ),
                        request,
                        None,
                        |value| validate_choices(value, stream, 4, &expected_prompt, 12),
                        None,
                    )?;
                }
            }
        }
        for special in [false, true] {
            for chat in [false, true] {
                let request = with_input(
                    json!({"model": model, "add_special_tokens": special,
                           "return_token_ids": true, "max_tokens": 1, "temperature": 0}),
                    chat,
                );
                run.check(
                    &format!(
                        "add_special_tokens_{}_{}",
                        py::name(chat),
                        py::name(special)
                    ),
                    request,
                    None,
                    |value| {
                        let ids = if chat {
                            py::get(value, "prompt_token_ids")?
                        } else {
                            py::get(
                                &py::at(&py::item(value, "choices")?, 0)?,
                                "prompt_token_ids",
                            )?
                        };
                        need(truthy(&ids), "missing prompt IDs")
                    },
                    None,
                )?;
            }
        }
        if let Some(tokens) = &tokens {
            run.check(
                "detokenize",
                json!({"model": model, "tokens": tokens}),
                Some("/detokenize"),
                |value| {
                    need(
                        py::item(value, "prompt")? == "Hello",
                        "tokenizer round trip differs",
                    )
                },
                None,
            )?;
        }
        if let Some(tokens) = tokens.as_ref().filter(|tokens| truthy(tokens)) {
            let stop_id = py::at(tokens, 0).map_err(crashed)?;
            let mut bias = Map::new();
            bias.insert(repr::str(&stop_id), json!(100));
            run.check(
                "stop_token_ids",
                json!({"model": model, "prompt": "Hello", "max_tokens": 3, "temperature": 0,
                       "logit_bias": bias, "stop_token_ids": [stop_id]}),
                None,
                |value| {
                    let stopped = python_eq(
                        &py::item(&py::item(value, "usage")?, "completion_tokens")?,
                        &json!(1),
                    ) && py::item(&py::at(&py::item(value, "choices")?, 0)?, "text")?
                        == ""
                        && py::item(&py::at(&py::item(value, "choices")?, 0)?, "finish_reason")?
                            == "stop";
                    need(stopped, "token stop differs")
                },
                None,
            )?;
        }
        run.check(
            "generation_prompt_control",
            json!({"model": model, "messages": hello, "max_tokens": 1,
                   "add_generation_prompt": false, "temperature": 0}),
            None,
            |value| {
                need(
                    truthy(&py::get(value, "choices")?),
                    "generation prefix control failed",
                )
            },
            None,
        )?;
        let question = json!([{"role": "user", "content": "What is 17 times 23?"}]);
        let (thinking_key, thinking) = thinking_control(run.capture, true)?;
        run.check(
            "hidden_reasoning",
            json!({"model": model, "messages": question, thinking_key: thinking,
                   "include_reasoning": false, "max_tokens": 8, "temperature": 0}),
            None,
            |value| {
                let message = py::item(&py::at(&py::item(value, "choices")?, 0)?, "message")?;
                let reasoning = py::get(&message, "reasoning")?;
                let shown = truthy(&reasoning) || truthy(&py::get(&message, "reasoning_content")?);
                need(!shown, "reasoning was not suppressed")
            },
            None,
        )?;
        run.check(
            "ignore_eos",
            json!({"model": model, "prompt": "Hello", "max_tokens": 3, "ignore_eos": true,
                   "skip_special_tokens": false, "temperature": 0, "stop_token_ids": []}),
            None,
            |value| {
                let bounded =
                    python_eq(
                        &py::item(&py::item(value, "usage")?, "completion_tokens")?,
                        &json!(3),
                    ) && py::item(&py::at(&py::item(value, "choices")?, 0)?, "finish_reason")?
                        == "length";
                need(bounded, "ignore_eos did not reach the length bound")
            },
            None,
        )?;
        run.check(
            "nullable_flags",
            json!({"model": model, "messages": hello, "max_tokens": 1, "temperature": 0,
                   "stream": null, "logprobs": null, "top_logprobs": 0}),
            None,
            |value| {
                need(
                    truthy(&py::get(value, "choices")?),
                    "nullable flags were refused",
                )
            },
            None,
        )?;
        run.check(
            "unknown_extension",
            json!({"model": model, "prompt": "Hello", "max_tokens": 1, "temperature": 0,
                   "parity_future_extension": true}),
            None,
            |value| {
                need(
                    truthy(&py::get(value, "choices")?),
                    "unknown extension was not ignored",
                )
            },
            None,
        )?;
        run.check(
            "sampling_extensions",
            json!({"model": model, "prompt": "Hello", "max_tokens": 3, "min_tokens": 2,
                   "temperature": 0, "seed": -1, "top_k": -1, "user": "api-qualification"}),
            None,
            |value| need(two_or_three(value)?, "minimum token bound differs"),
            None,
        )?;
        run.check(
            "chat_token_limit_precedence",
            json!({"model": model, "messages": hello, "max_tokens": 1,
                   "max_completion_tokens": 3, "min_tokens": 2, "temperature": 0}),
            None,
            |value| {
                need(
                    two_or_three(value)?,
                    "max_completion_tokens did not take precedence",
                )
            },
            None,
        )?;
        run.check(
            "continuous_usage",
            json!({"model": model, "prompt": "Hello", "max_tokens": 3, "temperature": 0,
                   "stream": true,
                   "stream_options": {"include_usage": true, "continuous_usage_stats": true}}),
            None,
            validate_continuous_usage,
            None,
        )?;
        let stream = run.check(
            "stream_usage",
            json!({"model": model, "messages": [{"role": "user", "content": "Say hello."}],
                   "stream": true, "stream_options": {"include_usage": true},
                   "max_tokens": 24, "temperature": 0}),
            None,
            validate_stream_usage,
            None,
        )?;
        run.check(
            "stop",
            json!({"model": model,
                   "messages": [{"role": "user",
                                 "content": "Write a short sentence ending with END."}],
                   "stop": ["END"], "max_tokens": 48, "temperature": 0}),
            None,
            |value| {
                let message = py::item(&py::at(&py::item(value, "choices")?, 0)?, "message")?;
                let content = py::get(&message, "content")?;
                let content = if truthy(&content) { content } else { json!("") };
                need(
                    !py::contains(&content, &json!("END"))?,
                    "stop string leaked",
                )
            },
            None,
        )?;
        run.check(
            "completion_logprobs",
            json!({"model": model, "prompt": "Paris is", "max_tokens": 5, "temperature": 0,
                   "logprobs": 5}),
            None,
            |value| {
                let sample = ["text_offset", "token_logprobs", "tokens", "top_logprobs"]
                    .map(|key| json!(key));
                let served = py::item(&py::at(&py::item(value, "choices")?, 0)?, "logprobs")?;
                let served = py::set(&served)?;
                need(
                    sample
                        .iter()
                        .all(|key| served.iter().any(|other| python_eq(key, other))),
                    "completion logprob shape differs",
                )
            },
            None,
        )?;
        let reasoning_field = capture::reasoning_field(run.capture)?;
        for thinking in [true, false] {
            let (thinking_key, control) = thinking_control(run.capture, thinking)?;
            let body = json!({"model": model, "messages": question, thinking_key: control,
                              "max_tokens": if mechanics { 8 } else { 128 }, "temperature": 0});
            if !thinking && !mechanics && !capture::thinking_can_be_disabled(run.capture)? {
                // A frontend that always reasons must refuse the disabled switch.
                run.refused("reasoning_False", body)?;
                continue;
            }
            run.check(
                &format!("reasoning_{}", py::name(thinking)),
                body,
                None,
                |value| {
                    let message = py::item(&py::at(&py::item(value, "choices")?, 0)?, "message")?;
                    if !mechanics {
                        need(
                            truthy(&py::get(&message, &reasoning_field)?) == thinking,
                            "reasoning mode mismatch",
                        )?;
                    }
                    Ok(())
                },
                None,
            )?;
        }
        let from_case = |name: &str| {
            py::item(&tool_case, name).map_err(|error| {
                KindError::Insufficient(format!(
                    "{}: the tools-00 conversation has no {error}",
                    path.display()
                ))
            })
        };
        let (tools, messages) = if mechanics {
            (
                json!([{"type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}}}]),
                json!([{"role": "user", "content": "Paris?"}]),
            )
        } else {
            (from_case("tools")?, from_case("messages")?)
        };
        let tool_stream = json!({"model": model, "messages": messages, "tools": tools,
                                 "tool_choice": "auto", "stream": true,
                                 "max_tokens": if mechanics { 1 } else { 256 },
                                 "temperature": 0});
        let mut whole = tool_stream.clone();
        whole["stream"] = json!(false);
        run.check_with(
            "stream_tool_arguments",
            tool_stream,
            None,
            |value, client| {
                validate_tool_stream(value, mechanics, || {
                    client
                        .json("/v1/chat/completions", Some(&whole))
                        .map(|(value, _)| value)
                        .map_err(|error| error.to_string())
                })
            },
            None,
        )?;
        let first = run.check(
            "tool_call",
            json!({"model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
                   "max_tokens": if mechanics { 1 } else { 128 }, "temperature": 0}),
            None,
            |value| {
                let called = mechanics || {
                    let choice = py::at(&py::item(value, "choices")?, 0)?;
                    py::item(&choice, "finish_reason")? == "tool_calls"
                        && truthy(&py::get(&py::item(&choice, "message")?, "tool_calls")?)
                };
                need(called, "model did not call the tool")
            },
            None,
        )?;
        if let Some(first) = first.filter(|_| !mechanics) {
            let calls = py::item(&first, "choices")
                .and_then(|choices| py::at(&choices, 0))
                .and_then(|choice| py::item(&choice, "message"))
                .and_then(|message| py::item(&message, "tool_calls"))
                .map_err(crashed)?;
            let id = py::at(&calls, 0)
                .and_then(|call| py::item(&call, "id"))
                .map_err(crashed)?;
            let messages = py::add(
                &messages,
                &json!([{"role": "assistant", "content": null, "tool_calls": calls},
                        {"role": "tool", "tool_call_id": id,
                         "content": "{\"temperature\":18,\"condition\":\"cloudy\"}"}]),
            )
            .map_err(crashed)?;
            // Thinking stays on, as users get it by default. Its reasoning can
            // take more than 128 tokens before the answer, so the answer has
            // more.
            run.check(
                "tool_result",
                json!({"model": model, "messages": messages, "tools": tools,
                       "max_tokens": tool_result_max_tokens, "temperature": 0}),
                None,
                validate_tool_result,
                Some(tool_result_facts),
            )?;
        }
        run.metrics()?;
        let stream_text = match stream.filter(truthy) {
            Some(events) => json!(response_text(&events).map_err(crashed)?),
            None => Value::Null,
        };
        let skipped = if mechanics {
            json!(["model reasoning and tool round trip (fixture context is 128 tokens)"])
        } else {
            json!([])
        };
        let mut result = Map::new();
        result.insert("passed".into(), json!(run.errors.is_empty()));
        result.insert("skipped".into(), skipped);
        result.insert("errors".into(), Value::Array(run.errors));
        result.insert("rows".into(), Value::Array(run.rows));
        result.insert("stream_text".into(), stream_text);
        Ok(receipt("api", run.capture, result))
    }

    /// One row per api.v1 case, in order, each with a boolean verdict
    /// (`kinds.Api.verify`).
    fn verify(
        &self,
        value: &Value,
        _reference: Option<&Value>,
        _check: &Check,
    ) -> Result<Vec<String>, KindError> {
        let rows = get(value, "rows").and_then(Value::as_array);
        require(
            rows.is_some_and(|rows| {
                rows.iter()
                    .all(|row| matches!(get(row, "passed"), Some(Value::Bool(_))))
            }),
            || "no case rows".into(),
        )?;
        let rows = rows.expect("checked");
        require(
            rows.len() == API_CASES.len()
                && rows
                    .iter()
                    .zip(API_CASES)
                    .all(|(row, case)| get(row, "name").and_then(Value::as_str) == Some(case)),
            || {
                format!(
                    "the rows are not the {} api.v1 cases in order",
                    API_CASES.len()
                )
            },
        )?;
        Ok(vec!["rows".into()])
    }

    /// `passed` is recomputed from the case rows (`kinds.Api.score`).
    fn score(
        &self,
        value: &Value,
        _reference: Option<&Value>,
        check: &Check,
        mechanics: bool,
    ) -> Result<Map<String, Value>, KindError> {
        let mut metrics = default_score(self, value, check, mechanics)?;
        if let Some(rows) = get(value, "rows").and_then(Value::as_array) {
            let mut passed = true;
            for row in rows {
                if !row.is_object() {
                    return Err(invalid(format!(
                        "'{}' object has no attribute 'get'",
                        repr::type_name(row)
                    )));
                }
                if row.get("passed") != Some(&Value::Bool(true)) {
                    passed = false;
                    break;
                }
            }
            metrics.insert("passed".into(), Value::Bool(passed));
        }
        Ok(metrics)
    }
}

/// `api.TOOL_RESULT_MAX_TOKENS`: the tool round trip's final answer, after
/// the model's default reasoning.
const TOOL_RESULT_MAX_TOKENS: u64 = 1024;

/// Why a case failed: Python's `str()` of what the case raised.
type Failure = String;

/// The facts a case's row records from its response.
type Facts = fn(&Value) -> Result<Map<String, Value>, Failure>;

/// Fails the case unless `condition` holds (`api.require`).
fn need(condition: bool, message: &str) -> Result<(), Failure> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// The cases run so far: their rows and the errors of those that failed.
struct Run<'c, 'a> {
    capture: &'c mut Capture<'a>,
    rows: Vec<Value>,
    errors: Vec<Value>,
}

impl Run<'_, '_> {
    /// One case (`api.main.check`): sends `body` to `path` (chat or
    /// completions by default), streamed when the body says so, and
    /// validates the response. A passed case returns its response; a failed
    /// one records its error with the rejected response. `details` maps a
    /// response to facts its row records, whether the case passes or not.
    /// A streamed response is its events as [seconds, event] pairs.
    fn check(
        &mut self,
        name: &str,
        body: Value,
        path: Option<&str>,
        validate: impl FnOnce(&Value) -> Result<(), Failure>,
        details: Option<Facts>,
    ) -> Result<Option<Value>, KindError> {
        self.check_with(name, body, path, |value, _| validate(value), details)
    }

    /// A case whose validation may send further requests with the client.
    fn check_with(
        &mut self,
        name: &str,
        body: Value,
        path: Option<&str>,
        validate: impl FnOnce(&Value, &mut Client) -> Result<(), Failure>,
        details: Option<Facts>,
    ) -> Result<Option<Value>, KindError> {
        let path = path.unwrap_or(if body.get("messages").is_some() {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        });
        capture::remaining(self.capture)?;
        let client = &mut *self.capture.client;
        let sent = if body.get("stream").is_some_and(truthy) {
            client.stream(path, &body).map(|(events, seconds)| {
                let events = events
                    .into_iter()
                    .map(|(at, event)| json!([at, event]))
                    .collect();
                (Value::Array(events), seconds)
            })
        } else {
            client.json(path, Some(&body))
        };
        let (value, error) = match sent {
            Ok((value, latency)) => {
                let checked = details
                    .map_or_else(|| Ok(Map::new()), |details| details(&value))
                    .and_then(|facts| validate(&value, self.capture.client).map(|()| facts));
                match checked {
                    Ok(facts) => {
                        let mut row = Map::new();
                        row.insert("name".into(), json!(name));
                        row.insert("latency_seconds".into(), json!(latency));
                        row.insert("passed".into(), json!(true));
                        row.extend(facts);
                        self.rows.push(Value::Object(row));
                        return Ok(Some(value));
                    }
                    Err(error) => (Some(value), error),
                }
            }
            Err(error @ HttpError::Transport(_)) => return Err(capture::http_error(error)),
            Err(error) => (None, error.to_string()),
        };
        // Keep the response a case rejected: a model-output failure cannot
        // be explained without it.
        let value = value.filter(|value| !value.is_null());
        let mut entry = Map::new();
        entry.insert("name".into(), json!(name));
        entry.insert("error".into(), json!(error));
        if let Some(value) = &value {
            entry.insert("response".into(), value.clone());
        }
        self.errors.push(Value::Object(entry));
        let mut row = Map::new();
        row.insert("name".into(), json!(name));
        row.insert("passed".into(), json!(false));
        if let (Some(value), Some(details)) = (&value, details) {
            row.extend(details(value).unwrap_or_default());
        }
        self.rows.push(Value::Object(row));
        Ok(None)
    }

    /// A chat request the frontend must refuse with HTTP 400 (`api.main`'s
    /// `refused`): the row records the first 200 characters of the refusal.
    fn refused(&mut self, name: &str, body: Value) -> Result<(), KindError> {
        capture::remaining(self.capture)?;
        let failure = match self
            .capture
            .client
            .json("/v1/chat/completions", Some(&body))
        {
            Ok((value, _)) => {
                json!({"name": name, "error": "accepted a request the frontend refuses", "response": value})
            }
            Err(error @ HttpError::Transport(_)) => return Err(capture::http_error(error)),
            Err(error) => {
                let text = error.to_string();
                if text.starts_with("HTTP 400") {
                    let refused: String = text.chars().take(200).collect();
                    self.rows
                        .push(json!({"name": name, "passed": true, "refused": refused}));
                    return Ok(());
                }
                json!({"name": name, "error": format!("expected HTTP 400, got {text}")})
            }
        };
        self.errors.push(failure);
        self.rows.push(json!({"name": name, "passed": false}));
        Ok(())
    }

    /// The `metrics_compatibility` case: `GET /metrics` read as text
    /// (`api.validate_metrics`).
    fn metrics(&mut self) -> Result<(), KindError> {
        capture::remaining(self.capture)?;
        let checked = match self.capture.client.request("GET", "/metrics", None) {
            Ok(response) => decode(&response.body).and_then(validate_metrics),
            Err(error @ HttpError::Transport(_)) => return Err(capture::http_error(error)),
            Err(error) => Err(error.to_string()),
        };
        let name = "metrics_compatibility";
        if let Err(error) = &checked {
            self.errors.push(json!({"name": name, "error": error}));
        }
        self.rows
            .push(json!({"name": name, "passed": checked.is_ok()}));
        Ok(())
    }
}

/// The events of a streamed response, without their times.
fn events(value: &Value) -> Vec<Value> {
    value
        .as_array()
        .map(|pairs| pairs.iter().map(|pair| pair[1].clone()).collect())
        .unwrap_or_default()
}

fn two_or_three(value: &Value) -> Result<bool, Failure> {
    let count = py::item(&py::item(value, "usage")?, "completion_tokens")?;
    Ok(py::compare(&json!(2), "<=", &count)? && py::compare(&count, "<=", &json!(3))?)
}

/// `api.validate_choices`: `count` choices finishing at the length bound,
/// usage of `prompt_tokens` and `completion_tokens`; a stream ends with a
/// usage-only chunk and has one response ID.
fn validate_choices(
    value: &Value,
    stream: bool,
    count: i64,
    prompt_tokens: &Value,
    completion_tokens: i64,
) -> Result<(), Failure> {
    let (choices, usage) = if stream {
        let values = events(value);
        let mut choices = Vec::new();
        for event in &values {
            for choice in py::iter(&py::get_or(event, "choices", json!([]))?)? {
                if !py::get(&choice, "finish_reason")?.is_null() {
                    choices.push(choice);
                }
            }
        }
        let last = py::at(&json!(values), -1)?;
        let usage = py::item(&last, "usage")?;
        need(
            python_eq(&py::item(&last, "choices")?, &json!([])),
            "missing aggregate usage chunk",
        )?;
        let mut ids: Vec<Value> = Vec::new();
        for event in &values {
            let id = py::item(event, "id")?;
            py::hashable(&id)?;
            if !ids.iter().any(|seen| python_eq(seen, &id)) {
                ids.push(id);
            }
        }
        need(ids.len() == 1, "response IDs differ")?;
        (choices, usage)
    } else {
        let choices = py::item(value, "choices")?;
        let usage = py::item(value, "usage")?;
        (py::iter(&choices)?, usage)
    };
    let indexes = choices
        .iter()
        .map(|choice| py::item(choice, "index"))
        .collect::<Result<Vec<_>, _>>()?;
    let indexes = py::sorted(indexes)?;
    need(
        indexes.len() as i64 == count
            && (0..)
                .zip(&indexes)
                .all(|(at, index)| python_eq(index, &json!(at))),
        "choice indexes differ",
    )?;
    for choice in &choices {
        need(
            py::item(choice, "finish_reason")? == "length",
            "choice finish reason differs",
        )?;
    }
    need(
        python_eq(&py::item(&usage, "prompt_tokens")?, prompt_tokens),
        "prompt usage duplicated",
    )?;
    need(
        python_eq(
            &py::item(&usage, "completion_tokens")?,
            &json!(completion_tokens),
        ),
        "completion usage differs",
    )?;
    let total = py::item(&usage, "total_tokens")?;
    need(
        python_eq(&total, &py::add(prompt_tokens, &json!(completion_tokens))?),
        "total usage differs",
    )
}

/// `api.validate_token_ids`: prompt token IDs present, and as many
/// generated IDs as completion tokens, each a non-negative int.
fn validate_token_ids(value: &Value, stream: bool, chat: bool) -> Result<(), Failure> {
    let (ids, usage) = if stream {
        let values = events(value);
        let mut clean = true;
        for event in &values {
            if py::contains(event, &json!("error"))? {
                clean = false;
                break;
            }
        }
        need(
            clean,
            &format!("token stream failed: {}", repr::repr(&json!(values))),
        )?;
        let mut ids = Vec::new();
        let mut prompts = Vec::new();
        for event in &values {
            for choice in py::iter(&py::get_or(event, "choices", json!([]))?)? {
                let tokens = py::get(&choice, "token_ids")?;
                if truthy(&tokens) {
                    ids.extend(py::iter(&tokens)?);
                }
            }
        }
        for event in &values {
            if chat {
                prompts.push(py::get(event, "prompt_token_ids")?);
            } else {
                for choice in py::iter(&py::get_or(event, "choices", json!([]))?)? {
                    prompts.push(py::get(&choice, "prompt_token_ids")?);
                }
            }
        }
        need(prompts.iter().any(truthy), "missing prompt IDs")?;
        (json!(ids), py::item(&py::at(&json!(values), -1)?, "usage")?)
    } else {
        let first = || py::at(&py::item(value, "choices")?, 0);
        let ids = py::item(&first()?, "token_ids")?;
        let prompt = if chat {
            py::get(value, "prompt_token_ids")?
        } else {
            py::get(&first()?, "prompt_token_ids")?
        };
        need(truthy(&prompt), "missing prompt IDs")?;
        (ids, py::item(value, "usage")?)
    };
    let length = py::len(&ids)?;
    need(
        python_eq(&json!(length), &py::item(&usage, "completion_tokens")?),
        "generated token IDs differ from usage",
    )?;
    // Python's `isinstance(token, int) and token >= 0`: a bool is an int.
    let valid = py::iter(&ids)?.iter().all(|token| match token {
        Value::Bool(_) => true,
        Value::Number(number) => !number.is_f64() && number.as_u64().is_some(),
        _ => false,
    });
    need(valid, "invalid token IDs")
}

/// A streamed tool call as `api.validate_tool_stream` gathers it.
struct Call {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    /// Arguments arrived before the name.
    unnamed: bool,
    /// Arguments arrived with or after the finish.
    late: bool,
}

/// `api.validate_tool_stream`: no error event, every chunk's choice is
/// choice 0, and tool indexes count up from 0. Outside a mechanics run, a
/// call finishing `tool_calls`, each call with a unique ID and one name,
/// sent before its arguments, which are complete before the finish and
/// form a JSON object, and the calls' names and arguments equal those of
/// `unstreamed()`, the same request served whole. The arguments may arrive
/// in one delta or several.
fn validate_tool_stream(
    value: &Value,
    mechanics: bool,
    unstreamed: impl FnOnce() -> Result<Value, Failure>,
) -> Result<(), Failure> {
    let mut calls: Vec<Call> = Vec::new();
    let mut terminal = Value::Null;
    for event in events(value) {
        need(
            !py::contains(&event, &json!("error"))?,
            "tool stream failed",
        )?;
        for choice in py::iter(&py::get_or(&event, "choices", json!([]))?)? {
            need(
                integer(&py::get(&choice, "index")?) == Some(0),
                "tool stream has another choice",
            )?;
            let finish = py::get(&choice, "finish_reason")?;
            if truthy(&finish) {
                terminal = finish;
            }
            let delta = py::get_or(&choice, "delta", json!({}))?;
            for delta in py::iter(&py::get_or(&delta, "tool_calls", json!([]))?)? {
                let at = integer(&py::item(&delta, "index")?)
                    .and_then(|index| usize::try_from(index).ok())
                    .filter(|at| *at <= calls.len())
                    .ok_or_else(|| "invalid tool index".to_owned())?;
                if at == calls.len() {
                    calls.push(Call {
                        id: None,
                        name: None,
                        arguments: String::new(),
                        unnamed: false,
                        late: false,
                    });
                }
                match py::get(&delta, "id")? {
                    Value::Null => {}
                    Value::String(id) if !id.is_empty() => {
                        need(
                            calls.iter().all(|call| call.id.as_ref() != Some(&id))
                                && calls[at].id.is_none(),
                            "tool ID repeated",
                        )?;
                        calls[at].id = Some(id);
                    }
                    _ => return Err("invalid tool ID".into()),
                }
                let call = &mut calls[at];
                let function = py::get_or(&delta, "function", json!({}))?;
                match py::get(&function, "name")? {
                    Value::Null => {}
                    Value::String(name) if name.is_empty() => {}
                    Value::String(name) => {
                        need(call.name.is_none(), "tool name repeated")?;
                        call.name = Some(name);
                    }
                    _ => return Err("invalid tool name".into()),
                }
                match py::get(&function, "arguments")? {
                    Value::String(part) if !part.is_empty() => {
                        call.arguments.push_str(&part);
                        call.unnamed |= call.name.is_none();
                        call.late |= truthy(&terminal);
                    }
                    part if truthy(&part) => {
                        return Err(format!(
                            "can only concatenate str (not \"{}\") to str",
                            type_name(&part)
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    if !mechanics {
        need(
            !calls.is_empty() && terminal == "tool_calls",
            "missing streaming tool call",
        )?;
        for call in &calls {
            need(
                call.id.is_some() && call.name.is_some(),
                "missing tool identity",
            )?;
            need(!call.unnamed, "tool arguments came before the tool name")?;
            need(!call.late, "tool arguments came with or after the finish")?;
            need(
                json_object(&json!(call.arguments)).is_some(),
                "tool arguments are not a JSON object",
            )?;
        }
        let response = unstreamed()?;
        let message = py::item(&py::at(&py::item(&response, "choices")?, 0)?, "message")?;
        let whole = py::get(&message, "tool_calls")?;
        let whole = if truthy(&whole) {
            py::iter(&whole)?
        } else {
            Vec::new()
        };
        let streamed: Vec<Value> = calls
            .iter()
            .map(|call| json!([call.name, json_object(&json!(call.arguments))]))
            .collect();
        let mut served = Vec::new();
        for call in &whole {
            let function = py::item(call, "function")?;
            served.push(json!([
                py::item(&function, "name")?,
                json_object(&py::item(&function, "arguments")?)
            ]));
        }
        need(
            same_json(&json!(streamed), &json!(served)),
            "streamed tool calls differ from the unstreamed response",
        )?;
    }
    Ok(())
}

/// A JSON integer (not a float or a boolean) as Python's `type(value) is int`
/// reads it.
fn integer(value: &Value) -> Option<i128> {
    match value {
        Value::Number(number) if !number.is_f64() => number
            .as_i64()
            .map(i128::from)
            .or(number.as_u64().map(i128::from)),
        _ => None,
    }
}

/// `api.same_json`: JSON values equal with their types; an integer never
/// equals a float (1 is not 1.0) and a boolean never equals a number.
fn same_json(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| same_json(value, other)))
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_json(a, b))
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            a.as_f64() == b.as_f64()
        }
        (Value::Number(_), Value::Number(_)) => {
            integer(left).is_some() && integer(left) == integer(right)
        }
        _ => left == right,
    }
}

/// `api.json_object`: `text` parsed when it is a JSON string holding a
/// JSON object.
fn json_object(text: &Value) -> Option<Value> {
    text.as_str()
        .and_then(|text| json::parse(text.as_bytes()).ok())
        .filter(Value::is_object)
}

/// `api.validate_continuous_usage`: usage on every chunk, never
/// decreasing, with consistent totals, ending in a usage-only chunk.
fn validate_continuous_usage(value: &Value) -> Result<(), Failure> {
    let values = events(value);
    let last = match values.last() {
        Some(last) => python_eq(&py::get(last, "choices")?, &json!([])),
        None => false,
    };
    need(last, "missing final usage chunk")?;
    let mut previous = json!(0);
    for value in &values {
        let usage = py::get(value, "usage")?;
        need(usage.is_object(), "missing continuous usage")?;
        let count = py::item(&usage, "completion_tokens")?;
        need(py::compare(&count, ">=", &previous)?, "usage regressed")?;
        let total = py::item(&usage, "total_tokens")?;
        let sum = py::add(&py::item(&usage, "prompt_tokens")?, &count)?;
        need(python_eq(&total, &sum), "usage total differs")?;
        previous = count;
    }
    let usage = py::item(&py::at(&json!(values), -1)?, "usage")?;
    let details = py::get(&usage, "prompt_tokens_details")?;
    if !details.is_null() {
        let cached = py::item(&details, "cached_tokens")?;
        let valid = py::compare(&json!(0), "<=", &cached)?
            && py::compare(&cached, "<=", &py::item(&usage, "prompt_tokens")?)?;
        need(valid, "invalid cache usage")?;
    }
    Ok(())
}

/// The `stream_usage` case: a chat chunk and a usage-only chunk.
fn validate_stream_usage(value: &Value) -> Result<(), Failure> {
    let values = events(value);
    let mut chunk = false;
    for event in &values {
        if py::get(event, "object")? == "chat.completion.chunk" {
            chunk = true;
            break;
        }
    }
    need(chunk, "missing chat chunk")?;
    let mut usage = false;
    for event in &values {
        if python_eq(&py::get(event, "choices")?, &json!([]))
            && py::contains(event, &json!("usage"))?
        {
            usage = true;
            break;
        }
    }
    need(usage, "missing usage chunk")
}

/// The `tool_result` case: the answer mentions the tool's result.
fn validate_tool_result(value: &Value) -> Result<(), Failure> {
    let choice = py::at(&py::item(value, "choices")?, 0)?;
    let content = py::get(&py::item(&choice, "message")?, "content")?;
    let used = match content {
        Value::String(text) => {
            let text = text.to_lowercase();
            text.contains("18") || text.contains("cloudy")
        }
        content if truthy(&content) => {
            return Err(format!(
                "'{}' object has no attribute 'lower'",
                type_name(&content)
            ));
        }
        _ => false,
    };
    let limited = py::get(&choice, "finish_reason")? == "length";
    need(
        used,
        &format!(
            "final answer did not use tool result{}",
            if limited {
                " (length limit reached)"
            } else {
                ""
            }
        ),
    )
}

/// The `tool_result` row's facts: the finish reason and whether it is the
/// length bound.
fn tool_result_facts(value: &Value) -> Result<Map<String, Value>, Failure> {
    let finish = py::get(&py::at(&py::item(value, "choices")?, 0)?, "finish_reason")?;
    let mut facts = Map::new();
    facts.insert("finish_reason".into(), finish.clone());
    facts.insert("length_limited".into(), json!(finish == "length"));
    Ok(facts)
}

/// `api.validate_metrics`: the vLLM-compatible series with model and
/// engine labels and non-negative values, a finish reason label and the
/// histograms' `+Inf` buckets.
fn validate_metrics(text: &str) -> Result<(), Failure> {
    let samples: Vec<&str> = py::splitlines(text)
        .into_iter()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let histograms = [
        "vllm:time_to_first_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
    ];
    let mut names: Vec<String> = [
        "vllm:request_success_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ]
    .map(String::from)
    .to_vec();
    for name in histograms {
        for suffix in ["_bucket", "_sum", "_count"] {
            names.push(format!("{name}{suffix}"));
        }
    }
    for name in &names {
        let prefix = format!("{name}{{");
        let series: Vec<&str> = samples
            .iter()
            .copied()
            .filter(|line| line.starts_with(&prefix))
            .collect();
        need(
            !series.is_empty(),
            &format!("missing compatibility series {name}"),
        )?;
        for line in series {
            need(
                line.contains("model_name=\"") && line.contains("engine=\""),
                &format!("missing labels: {line}"),
            )?;
            let (_, value) = line
                .rsplit_once(' ')
                .ok_or_else(|| "list index out of range".to_owned())?;
            need(
                py::float(value)? >= 0.0,
                &format!("negative metric: {line}"),
            )?;
        }
    }
    need(
        samples.iter().any(|line| {
            line.starts_with("vllm:request_success_total{") && line.contains("finished_reason=\"")
        }),
        "missing finish reason label",
    )?;
    for name in histograms {
        let prefix = format!("{name}_bucket{{");
        need(
            samples
                .iter()
                .any(|line| line.starts_with(&prefix) && line.contains("le=\"+Inf\"")),
            &format!("missing infinite bucket {name}"),
        )?;
    }
    Ok(())
}

/// Python's `bytes.decode()`: strict UTF-8.
fn decode(bytes: &[u8]) -> Result<&str, Failure> {
    std::str::from_utf8(bytes).map_err(|error| {
        let at = error.valid_up_to();
        let reason = match error.error_len() {
            None => "unexpected end of data",
            Some(_) if matches!(bytes[at], 0x80..=0xc1 | 0xf5..=0xff) => "invalid start byte",
            Some(_) => "invalid continuation byte",
        };
        format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {at}: {reason}",
            bytes[at]
        )
    })
}

/// The streamed chat text (`common.response_text(events, chat=True)`).
fn response_text(value: &Value) -> Result<String, Failure> {
    let mut text = String::new();
    let mut parts = 0;
    for event in events(value) {
        for choice in py::iter(&py::get_or(&event, "choices", json!([]))?)? {
            match py::get(&py::get_or(&choice, "delta", json!({}))?, "content")? {
                Value::String(part) if !part.is_empty() => text.push_str(&part),
                part if truthy(&part) => {
                    return Err(format!(
                        "sequence item {parts}: expected str instance, {} found",
                        type_name(&part)
                    ));
                }
                _ => continue,
            }
            parts += 1;
        }
    }
    Ok(text)
}

/// Python's operations on JSON values as `api.py` applies them to
/// responses, failing with the message Python's exception carries.
mod py {
    use std::cmp::Ordering;

    use serde_json::{Value, json};

    use super::Failure;
    use crate::json::{python_cmp, python_eq};
    use crate::repr::{self, type_name};

    /// `True` or `False`, as an f-string writes a bool.
    pub fn name(flag: bool) -> &'static str {
        if flag { "True" } else { "False" }
    }

    fn unsubscriptable(value: &Value) -> Failure {
        format!("'{}' object is not subscriptable", type_name(value))
    }

    /// `value[key]`.
    pub fn item(value: &Value, key: &str) -> Result<Value, Failure> {
        match value {
            Value::Object(map) => map.get(key).cloned().ok_or_else(|| repr::repr(&json!(key))),
            Value::Array(_) => Err("list indices must be integers or slices, not str".into()),
            Value::String(_) => Err("string indices must be integers, not 'str'".into()),
            other => Err(unsubscriptable(other)),
        }
    }

    /// `value[index]`, a negative index counting from the end.
    pub fn at(value: &Value, index: i64) -> Result<Value, Failure> {
        let position = |length: usize| {
            let at = if index < 0 {
                index + length as i64
            } else {
                index
            };
            usize::try_from(at).ok().filter(|at| *at < length)
        };
        match value {
            Value::Array(items) => position(items.len())
                .map(|at| items[at].clone())
                .ok_or_else(|| "list index out of range".into()),
            Value::String(text) => {
                let characters: Vec<char> = text.chars().collect();
                position(characters.len())
                    .map(|at| json!(characters[at].to_string()))
                    .ok_or_else(|| "string index out of range".into())
            }
            // JSON keys are strings: an int key is never present.
            Value::Object(_) => Err(index.to_string()),
            other => Err(unsubscriptable(other)),
        }
    }

    /// `value.get(key, default)`.
    pub fn get_or(value: &Value, key: &str, default: Value) -> Result<Value, Failure> {
        match value {
            Value::Object(map) => Ok(map.get(key).cloned().unwrap_or(default)),
            other => Err(format!(
                "'{}' object has no attribute 'get'",
                type_name(other)
            )),
        }
    }

    /// `value.get(key)`: null when absent.
    pub fn get(value: &Value, key: &str) -> Result<Value, Failure> {
        get_or(value, key, Value::Null)
    }

    /// Iteration: a list's items, a dict's keys, a string's characters.
    pub fn iter(value: &Value) -> Result<Vec<Value>, Failure> {
        match value {
            Value::Array(items) => Ok(items.clone()),
            Value::Object(map) => Ok(map.keys().map(|key| json!(key)).collect()),
            Value::String(text) => Ok(text.chars().map(|c| json!(c.to_string())).collect()),
            other => Err(format!("'{}' object is not iterable", type_name(other))),
        }
    }

    /// `set(value)`, its items in first-seen order.
    pub fn set(value: &Value) -> Result<Vec<Value>, Failure> {
        let mut items: Vec<Value> = Vec::new();
        for item in iter(value)? {
            hashable(&item)?;
            if !items.iter().any(|seen| python_eq(seen, &item)) {
                items.push(item);
            }
        }
        Ok(items)
    }

    /// `len(value)`.
    pub fn len(value: &Value) -> Result<usize, Failure> {
        match value {
            Value::String(text) => Ok(text.chars().count()),
            Value::Array(items) => Ok(items.len()),
            Value::Object(map) => Ok(map.len()),
            other => Err(format!(
                "object of type '{}' has no len()",
                type_name(other)
            )),
        }
    }

    /// Refuses a value Python cannot hash: a list or a dict.
    pub fn hashable(value: &Value) -> Result<(), Failure> {
        match value {
            Value::Array(_) | Value::Object(_) => {
                Err(format!("unhashable type: '{}'", type_name(value)))
            }
            _ => Ok(()),
        }
    }

    /// `item in container`.
    pub fn contains(container: &Value, item: &Value) -> Result<bool, Failure> {
        match container {
            Value::Object(map) => {
                hashable(item)?;
                Ok(item.as_str().is_some_and(|key| map.contains_key(key)))
            }
            Value::Array(items) => Ok(items.iter().any(|other| python_eq(other, item))),
            Value::String(text) => match item {
                Value::String(part) => Ok(text.contains(part.as_str())),
                other => Err(format!(
                    "'in <string>' requires string as left operand, not {}",
                    type_name(other)
                )),
            },
            other => Err(format!(
                "argument of type '{}' is not iterable",
                type_name(other)
            )),
        }
    }

    /// The order of two values Python compares: numbers (bools count as
    /// ints), strings, and lists by their first unequal items; `None` for
    /// unordered floats. The types of the pair that does not compare
    /// otherwise.
    fn order(
        left: &Value,
        right: &Value,
    ) -> Result<Option<Ordering>, (&'static str, &'static str)> {
        match (left, right) {
            (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => {
                Ok(python_cmp(left, right))
            }
            (Value::String(a), Value::String(b)) => Ok(Some(a.cmp(b))),
            (Value::Array(a), Value::Array(b)) => {
                for (a, b) in a.iter().zip(b) {
                    if !python_eq(a, b) {
                        return order(a, b);
                    }
                }
                Ok(Some(a.len().cmp(&b.len())))
            }
            _ => Err((type_name(left), type_name(right))),
        }
    }

    /// `left <op> right` for `<`, `<=` and `>=`.
    pub fn compare(left: &Value, op: &str, right: &Value) -> Result<bool, Failure> {
        let order = order(left, right).map_err(|(left, right)| {
            format!("'{op}' not supported between instances of '{left}' and '{right}'")
        })?;
        Ok(order.is_some_and(|order| match op {
            "<" => order.is_lt(),
            "<=" => order.is_le(),
            _ => order.is_ge(),
        }))
    }

    /// `sorted(items)`, by `<`.
    pub fn sorted(mut items: Vec<Value>) -> Result<Vec<Value>, Failure> {
        for next in 1..items.len() {
            let mut at = next;
            while at > 0 && compare(&items[at], "<", &items[at - 1])? {
                items.swap(at, at - 1);
                at -= 1;
            }
        }
        Ok(items)
    }

    enum Number {
        Int(i128),
        Float(f64),
    }

    fn number(value: &Value) -> Option<Number> {
        match value {
            Value::Bool(flag) => Some(Number::Int(i128::from(*flag))),
            Value::Number(number) if number.is_f64() => number.as_f64().map(Number::Float),
            Value::Number(number) => number
                .as_i64()
                .map(i128::from)
                .or_else(|| number.as_u64().map(i128::from))
                .map(Number::Int),
            _ => None,
        }
    }

    fn as_float(number: Number) -> f64 {
        match number {
            Number::Int(value) => value as f64,
            Number::Float(value) => value,
        }
    }

    fn int(value: i128) -> Value {
        if let Ok(value) = u64::try_from(value) {
            json!(value)
        } else if let Ok(value) = i64::try_from(value) {
            json!(value)
        } else {
            json!(value as f64)
        }
    }

    /// `left + right`.
    pub fn add(left: &Value, right: &Value) -> Result<Value, Failure> {
        match (number(left), number(right)) {
            (Some(Number::Int(a)), Some(Number::Int(b))) => return Ok(int(a + b)),
            (Some(a), Some(b)) => return Ok(json!(as_float(a) + as_float(b))),
            _ => {}
        }
        match (left, right) {
            (Value::String(a), Value::String(b)) => Ok(json!(format!("{a}{b}"))),
            (Value::Array(a), Value::Array(b)) => Ok(json!([a.as_slice(), b.as_slice()].concat())),
            (Value::String(_), other) => Err(format!(
                "can only concatenate str (not \"{}\") to str",
                type_name(other)
            )),
            (Value::Array(_), other) => Err(format!(
                "can only concatenate list (not \"{}\") to list",
                type_name(other)
            )),
            _ => Err(format!(
                "unsupported operand type(s) for +: '{}' and '{}'",
                type_name(left),
                type_name(right)
            )),
        }
    }

    /// `2 * value`.
    pub fn double(value: &Value) -> Result<Value, Failure> {
        match (number(value), value) {
            (Some(Number::Int(a)), _) => Ok(int(2 * a)),
            (Some(Number::Float(a)), _) => Ok(json!(2.0 * a)),
            (None, Value::String(text)) => Ok(json!(text.repeat(2))),
            (None, Value::Array(items)) => Ok(json!([items.as_slice(), items.as_slice()].concat())),
            (None, other) => Err(format!(
                "unsupported operand type(s) for *: 'int' and '{}'",
                type_name(other)
            )),
        }
    }

    /// `str.splitlines()`.
    pub fn splitlines(text: &str) -> Vec<&str> {
        let mut lines = Vec::new();
        let mut start = 0;
        let mut characters = text.char_indices().peekable();
        while let Some((at, character)) = characters.next() {
            if matches!(
                character,
                '\n' | '\r'
                    | '\x0b'
                    | '\x0c'
                    | '\x1c'
                    | '\x1d'
                    | '\x1e'
                    | '\u{85}'
                    | '\u{2028}'
                    | '\u{2029}'
            ) {
                lines.push(&text[start..at]);
                start = at + character.len_utf8();
                if character == '\r' && characters.peek().is_some_and(|(_, next)| *next == '\n') {
                    characters.next();
                    start += 1;
                }
            }
        }
        if start < text.len() {
            lines.push(&text[start..]);
        }
        lines
    }

    /// `float(text)`: surrounding whitespace, a sign, `inf`, `infinity` and
    /// `nan` in any case, and single underscores between digits.
    pub fn float(text: &str) -> Result<f64, Failure> {
        let refused = || {
            format!(
                "could not convert string to float: {}",
                repr::repr(&json!(text))
            )
        };
        let trimmed =
            text.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
        let bytes = trimmed.as_bytes();
        let digit = |at: Option<usize>| {
            at.and_then(|at| bytes.get(at))
                .is_some_and(u8::is_ascii_digit)
        };
        let grouped = bytes
            .iter()
            .enumerate()
            .all(|(at, byte)| *byte != b'_' || (digit(at.checked_sub(1)) && digit(Some(at + 1))));
        if !grouped {
            return Err(refused());
        }
        trimmed.replace('_', "").parse().map_err(|_| refused())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::{Command, Stdio};

    use serde_json::{Value, json};

    use super::*;

    /// A streamed response of `events`.
    fn stream(events: Value) -> Value {
        json!(
            events
                .as_array()
                .unwrap()
                .iter()
                .map(|event| json!([0.5, event]))
                .collect::<Vec<_>>()
        )
    }

    /// Validator calls on malformed and well-formed responses: the
    /// `api.py` function and its arguments.
    fn calls() -> Vec<(&'static str, Value)> {
        let usage = json!({"prompt_tokens": 1, "completion_tokens": 9, "total_tokens": 10});
        let choice = |index: Value| json!({"index": index, "finish_reason": "length"});
        let choices = |indexes: Value| -> Value {
            json!(
                indexes
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|index| choice(index.clone()))
                    .collect::<Vec<_>>()
            )
        };
        let done = json!({"id": "a", "choices": [], "usage": usage});
        let mut calls = vec![
            ("validate_choices", json!([{}, false, 3, 1, 9])),
            (
                "validate_choices",
                json!([{"choices": null, "usage": usage}, false, 3, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([2, 0, 1])), "usage": usage}, false, 3, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([2.0, true, 0])), "usage": usage}, false, 3, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([1, "0"])), "usage": usage}, false, 2, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": [{"index": 0}], "usage": usage}, false, 1, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([0])), "usage": null}, false, 1, 1, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([0])), "usage": usage}, false, 1, null, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([0])), "usage": {"prompt_tokens": null, "completion_tokens": 9}}, false, 1, null, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([0])), "usage": {"prompt_tokens": "a", "completion_tokens": 9, "total_tokens": 1}}, false, 1, "a", 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": choices(json!([0])), "usage": {"prompt_tokens": 1.0, "completion_tokens": 9, "total_tokens": 10}}, false, 1, 1.0, 9]),
            ),
            (
                "validate_choices",
                json!([{"choices": "ab", "usage": usage}, false, 2, 1, 9]),
            ),
            ("validate_choices", json!([[], true, 1, 1, 9])),
            (
                "validate_choices",
                json!([
                    stream(json!([{"id": "a", "choices": [choice(json!(0))]}, done])),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([
                    stream(json!([{"id": "b", "choices": [choice(json!(0))]}, done])),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([
                    stream(json!([{"id": [1], "choices": []}, done])),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([stream(json!([[1], done])), true, 1, 1, 9]),
            ),
            (
                "validate_choices",
                json!([
                    stream(json!([{"id": "a", "choices": null}, done])),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([
                    stream(
                        json!([{"id": "a", "choices": [{"index": 0, "finish_reason": null}]}, done])
                    ),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([
                    stream(
                        json!([{"id": "a", "choices": [choice(json!(0))]}, {"id": "a", "choices": [1]}])
                    ),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
            (
                "validate_choices",
                json!([
                    stream(
                        json!([{"id": "a", "choices": [choice(json!(0))]}, {"id": "a", "choices": [], "usage": usage, "x": 1}])
                    ),
                    true,
                    1,
                    1,
                    9
                ]),
            ),
        ];
        let token = |ids: Value, prompt: Value| {
            json!({"choices": [{"token_ids": ids, "prompt_token_ids": prompt}], "prompt_token_ids": prompt,
                   "usage": {"completion_tokens": 3}})
        };
        calls.extend([
            ("validate_token_ids", json!([token(json!([1, 2, 3]), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!([1, true, 3]), json!([5])), false, true])),
            ("validate_token_ids", json!([token(json!([1, -2, 3]), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!([1, 2.0, 3]), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!("abc"), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!(null), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!({"a": 1, "b": 2, "c": 3}), json!([5])), false, false])),
            ("validate_token_ids", json!([token(json!([1, 2, 3]), json!([])), false, false])),
            ("validate_token_ids", json!([{"choices": []}, false, false])),
            ("validate_token_ids", json!([{"choices": "x"}, false, true])),
            ("validate_token_ids", json!([stream(json!([{"error": {"message": "it's \"bad\""}}])), true, false])),
            ("validate_token_ids", json!([stream(json!([{"choices": [{"token_ids": [1], "prompt_token_ids": [5]}]},
                                                        {"choices": [], "usage": {"completion_tokens": 1}}])), true, false])),
            ("validate_token_ids", json!([stream(json!([{"prompt_token_ids": [5], "choices": [{"token_ids": null}]},
                                                        {"choices": [], "usage": {"completion_tokens": 0}}])), true, true])),
            ("validate_token_ids", json!([stream(json!([{"choices": [{"token_ids": 7}]}])), true, false])),
            ("validate_token_ids", json!([stream(json!([{"choices": [{"token_ids": [1]}]}])), true, true])),
            ("validate_token_ids", json!([stream(json!(["error"])), true, true])),
            ("validate_token_ids", json!([stream(json!([3])), true, true])),
        ]);
        let tool =
            |delta: Value| json!({"choices": [{"index": 0, "delta": {"tool_calls": [delta]}}]});
        let finish = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]});
        let first = json!({"index": 0, "id": "c", "function": {"name": "f", "arguments": "{"}});
        let rest = json!({"index": 0, "function": {"arguments": "}"}});
        // The response of the same request served whole, with these calls.
        let whole = |calls: Value| {
            let calls: Vec<Value> = calls
                .as_array()
                .unwrap()
                .iter()
                .map(|call| {
                    json!({"id": "w", "type": "function",
                                   "function": {"name": call[0], "arguments": call[1]}})
                })
                .collect();
            json!({"choices": [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"role": "assistant", "content": null,
                                            "tool_calls": calls}}]})
        };
        let empty = whole(json!([["f", "{}"]]));
        // The stream the server sends: the name with empty arguments, the
        // complete arguments when the call closes, then the finish. The
        // v0.1.0-alpha.1 quick gate failed it as "buffered whole".
        let named = json!({"index": 0, "id": "call-0", "type": "function",
                           "function": {"name": "get_weather", "arguments": ""}});
        let paris = |arguments: &str| json!({"index": 0, "function": {"arguments": arguments}});
        let closed = stream(json!([
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                          "finish_reason": null}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "The user wants the weather."}}]},
            tool(named.clone()),
            tool(paris("{\"city\": \"Paris\"}")),
            finish
        ]));
        let weather = |arguments: Value| whole(json!([["get_weather", arguments]]));
        let tool_stream = |events: Value, mechanics: bool, unstreamed: &Value| {
            (
                "validate_tool_stream",
                json!([stream(events), mechanics, unstreamed]),
            )
        };
        calls.extend([
            (
                "validate_tool_stream",
                json!([closed, false, weather(json!("{\"city\": \"Paris\"}"))]),
            ),
            // Equal JSON of other types differs: an integer is not a float or
            // a boolean, a string not a number.
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"n\": 1, \"b\": true}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"n\": 1, \"b\": true}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"n\": 1}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"n\": 1.0}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"n\": [1.0]}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"n\": [1e0]}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"b\": true}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"b\": 1}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"n\": 0}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"n\": false}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"n\": {\"m\": null}}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"n\": {\"m\": \"null\"}}"))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([
                    stream(json!([
                        tool(named.clone()),
                        tool(paris("{\"a\": 1, \"b\": 2}")),
                        finish
                    ])),
                    false,
                    weather(json!("{\"b\": 2, \"a\": 1}"))
                ]),
            ),
            // Equal as JSON, not as text.
            (
                "validate_tool_stream",
                json!([closed, false, weather(json!("{\"city\":\"Paris\"} "))]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, weather(json!("{\"city\": \"Rome\"}"))]),
            ),
            (
                "validate_tool_stream",
                json!([
                    closed,
                    false,
                    whole(json!([["other", "{\"city\": \"Paris\"}"]]))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, weather(json!("{\"city\": \"Paris\""))]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, weather(json!({"city": "Paris"}))]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, whole(json!([]))]),
            ),
            (
                "validate_tool_stream",
                json!([
                    closed,
                    false,
                    whole(json!([
                        ["get_weather", "{\"city\": \"Paris\"}"],
                        ["get_weather", "{}"]
                    ]))
                ]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, {"choices": []}]),
            ),
            (
                "validate_tool_stream",
                json!([closed, false, {"error": "x"}]),
            ),
            tool_stream(
                json!([tool(first.clone()), tool(rest.clone()), finish]),
                false,
                &empty,
            ),
            // Arguments before the name, with the finish and after it.
            tool_stream(
                json!([
                    tool(json!({"index": 0, "id": "c", "function": {"arguments": "{}"}})),
                    tool(json!({"index": 0, "function": {"name": "f"}})),
                    finish
                ]),
                false,
                &empty,
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    {"choices": [{"index": 0,
                                  "delta": {"tool_calls": [paris("{\"city\": \"Paris\"}")]},
                                  "finish_reason": "tool_calls"}]}
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(paris("{\"city\": ")),
                    finish,
                    tool(paris("\"Paris\"}"))
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            // Missing, malformed or non-object arguments.
            tool_stream(json!([tool(named.clone()), finish]), false, &empty),
            tool_stream(json!([tool(first.clone()), finish]), false, &empty),
            tool_stream(
                json!([tool(named.clone()), tool(paris("[1]")), finish]),
                false,
                &weather(json!("[1]")),
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(paris("{\"a\": 1, \"a\": 2}")),
                    finish
                ]),
                false,
                &weather(json!("{\"a\": 1, \"a\": 2}")),
            ),
            tool_stream(
                json!([tool(named.clone()), tool(paris("{\"a\": NaN}")), finish]),
                false,
                &weather(json!("{\"a\": NaN}")),
            ),
            tool_stream(
                json!([tool(first.clone()), tool(first.clone()), finish]),
                false,
                &empty,
            ),
            tool_stream(
                json!([tool(first.clone()), tool(rest.clone())]),
                false,
                &empty,
            ),
            tool_stream(
                json!([tool(first.clone()), tool(rest.clone())]),
                true,
                &empty,
            ),
            tool_stream(json!([tool(json!({"index": [0]}))]), true, &empty),
            tool_stream(json!([tool(json!({"id": "c"}))]), true, &empty),
            tool_stream(
                json!([tool(json!({"index": 0, "function": null}))]),
                true,
                &empty,
            ),
            tool_stream(
                json!([tool(json!({"index": 0, "function": {"arguments": 5}}))]),
                true,
                &empty,
            ),
            tool_stream(
                json!([
                    tool(
                        json!({"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}})
                    ),
                    tool(json!({"index": 0, "function": {"arguments": " "}})),
                    finish
                ]),
                false,
                &empty,
            ),
            tool_stream(
                json!([{"choices": [{"index": 0, "delta": {"tool_calls": null}}]}]),
                true,
                &empty,
            ),
            tool_stream(json!([{"error": "x"}]), true, &empty),
            // One choice: another choice index, or none, fails.
            tool_stream(
                json!([
                    tool(named.clone()),
                    {"choices": [{"index": 1,
                                  "delta": {"tool_calls": [paris("{\"city\": \"Paris\"}")]}}]},
                    finish
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            tool_stream(
                json!([{"choices": [{"index": 0.0, "delta": {}}]}]),
                true,
                &empty,
            ),
            tool_stream(json!([{"choices": [{"delta": {}}]}]), true, &empty),
            tool_stream(
                json!([{"choices": [{"index": false, "delta": {}}]}]),
                true,
                &empty,
            ),
            // Tool indexes count up from 0.
            tool_stream(
                json!([tool(
                    json!({"index": 1, "id": "c", "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            tool_stream(
                json!([tool(
                    json!({"index": 0.0, "id": "c", "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            tool_stream(
                json!([tool(
                    json!({"index": -1, "id": "c", "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            tool_stream(
                json!([tool(
                    json!({"index": true, "id": "c", "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            // Two calls pass; IDs must be unique, nonempty strings.
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(paris("{\"city\": \"Paris\"}")),
                    tool(json!({"index": 1, "id": "call-1",
                                "function": {"name": "get_weather", "arguments": "{}"}})),
                    finish
                ]),
                false,
                &whole(json!([
                    ["get_weather", "{\"city\": \"Paris\"}"],
                    ["get_weather", "{}"]
                ])),
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(paris("{\"city\": \"Paris\"}")),
                    tool(json!({"index": 1, "id": "call-0",
                                "function": {"name": "get_weather", "arguments": "{}"}})),
                    finish
                ]),
                false,
                &whole(json!([
                    ["get_weather", "{\"city\": \"Paris\"}"],
                    ["get_weather", "{}"]
                ])),
            ),
            tool_stream(
                json!([tool(
                    json!({"index": 0, "id": "", "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            tool_stream(
                json!([tool(
                    json!({"index": 0, "id": 7, "function": {"name": "f"}})
                )]),
                true,
                &empty,
            ),
            tool_stream(
                json!([
                    tool(json!({"index": 0, "function": {"name": "f", "arguments": "{}"}})),
                    finish
                ]),
                false,
                &empty,
            ),
            // A call is named once: a later name, before or after its
            // arguments, fails; an empty one is ignored.
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(json!({"index": 0, "function": {"name": "get_time"}})),
                    tool(paris("{\"city\": \"Paris\"}")),
                    finish
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(paris("{\"city\": \"Paris\"}")),
                    tool(json!({"index": 0, "function": {"name": "get_weather"}})),
                    finish
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            tool_stream(
                json!([
                    tool(named.clone()),
                    tool(json!({"index": 0, "function": {"name": "",
                                                         "arguments": "{\"city\": \"Paris\"}"}})),
                    finish
                ]),
                false,
                &weather(json!("{\"city\": \"Paris\"}")),
            ),
            tool_stream(
                json!([tool(
                    json!({"index": 0, "id": "c", "function": {"name": 1}})
                )]),
                true,
                &empty,
            ),
            tool_stream(json!([]), false, &empty),
        ]);
        let chunk = |prompt: Value, completion: Value, total: Value| json!({"choices": [{}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}});
        let last = |cached: Value| {
            json!({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
                                            "prompt_tokens_details": {"cached_tokens": cached}}})
        };
        calls.extend([
            (
                "validate_continuous_usage",
                json!([stream(json!([
                    chunk(json!(2), json!(1), json!(3)),
                    last(json!(0))
                ]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([
                    chunk(json!(2), json!(4), json!(6)),
                    last(json!(0))
                ]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([
                    chunk(json!(2), json!(1), json!(4)),
                    last(json!(0))
                ]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([
                    chunk(json!(2), json!("1"), json!(3)),
                    last(json!(0))
                ]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([
                    chunk(json!(2.5), json!(1), json!(3.5)),
                    last(json!(0))
                ]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([last(json!(3))]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([last(json!(-1))]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([last(json!("x"))]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([last(json!(null))]))]),
            ),
            (
                "validate_continuous_usage",
                json!([stream(json!([{"choices": []}]))]),
            ),
            ("validate_continuous_usage", json!([stream(json!([]))])),
            ("validate_continuous_usage", json!([stream(json!([[]]))])),
        ]);
        let labels = "{engine=\"0\",model_name=\"m\"}";
        let mut series: Vec<String> = [
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc",
        ]
        .iter()
        .map(|name| format!("{name}{labels} 1"))
        .collect();
        series.push(
            "vllm:request_success_total{engine=\"0\",model_name=\"m\",finished_reason=\"stop\"} 1"
                .into(),
        );
        for name in [
            "vllm:time_to_first_token_seconds",
            "vllm:inter_token_latency_seconds",
            "vllm:e2e_request_latency_seconds",
        ] {
            series.push(format!(
                "{name}_bucket{{engine=\"0\",model_name=\"m\",le=\"+Inf\"}} 1"
            ));
            series.push(format!("{name}_sum{labels} 1"));
            series.push(format!("{name}_count{labels} 1"));
        }
        let text = series.join("\n");
        for (from, to) in [
            ("", ""),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} 1_0",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} 1__0",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} NaN",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} +Inf",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} -inf",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} 1e-3\t",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} x'",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"}\t1",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} ",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{model_name=\"m\"} 1",
            ),
            (
                "perc{engine=\"0\",model_name=\"m\"} 1",
                "perc{engine=\"0\",model_name=\"m\"} 1\u{2028}vllm:kv_cache_usage_perc{x} 1",
            ),
            ("finished_reason", "finish_reason"),
            ("le=\"+Inf\"", "le=\"inf\""),
            ("vllm:num_requests_waiting{", "vllm:num_requests_waiting {"),
            ("\n", "\r\n# comment\n\n"),
        ] {
            calls.push(("validate_metrics", json!([text.replace(from, to)])));
        }
        calls
    }

    fn rust(name: &str, args: &Value) -> Result<(), Failure> {
        let int = |at: usize| args[at].as_i64().unwrap();
        let flag = |at: usize| args[at].as_bool().unwrap();
        match name {
            "validate_choices" => validate_choices(&args[0], flag(1), int(2), &args[3], int(4)),
            "validate_token_ids" => validate_token_ids(&args[0], flag(1), flag(2)),
            "validate_tool_stream" => {
                validate_tool_stream(&args[0], flag(1), || Ok(args[2].clone()))
            }
            "validate_continuous_usage" => validate_continuous_usage(&args[0]),
            "validate_metrics" => validate_metrics(args[0].as_str().unwrap()),
            _ => unreachable!("{name}"),
        }
    }

    /// Runs each call of `api.py`'s validators: `str()` of what it raised,
    /// or null. A tool stream's third argument is the whole response its
    /// `unstreamed` returns.
    const VALIDATE: &str = "
import json
import api
results = []
for name, args in json.load(sys.stdin):
    if name == 'validate_tool_stream':
        args[2] = (lambda whole: lambda: whole)(args[2])
    try:
        getattr(api, name)(*args)
        results.append(None)
    except Exception as error:
        results.append(str(error))
json.dump(results, sys.stdout)
";

    #[test]
    fn validators_fail_where_and_as_python_does() {
        let calls = calls();
        let tools = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/qual");
        let child = Command::new("python3")
            .arg("-B")
            .arg("-c")
            .arg(format!(
                "import sys\nsys.path.insert(0, {tools:?})\n{VALIDATE}"
            ))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipped: python3 is missing");
                return;
            }
            Err(error) => panic!("python3: {error}"),
        };
        let input = serde_json::to_vec(&calls).unwrap();
        child.stdin.take().unwrap().write_all(&input).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "python3 failed");
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), calls.len());
        let mut failures = Vec::new();
        for ((name, args), expected) in calls.iter().zip(&expected) {
            let actual = match rust(name, args) {
                Ok(()) => Value::Null,
                Err(error) => json!(error),
            };
            if &actual != expected {
                failures.push(format!("{name} {args}: {actual} != {expected}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert!(expected.iter().any(Value::is_null));
    }

    #[test]
    fn python_floats_and_lines_read_as_python_reads_them() {
        for (text, value) in [
            ("1_000.5", 1000.5),
            (" 2 ", 2.0),
            ("+Infinity", f64::INFINITY),
            ("-0", -0.0),
        ] {
            assert_eq!(py::float(text), Ok(value), "{text}");
        }
        assert!(py::float("nan").unwrap().is_nan());
        for text in ["_1", "1_", "1__0", "1_.5", "", "0x1"] {
            assert!(py::float(text).is_err(), "{text}");
        }
        assert_eq!(
            py::splitlines("a\r\nb\rc\n\nd\x0be\n"),
            ["a", "b", "c", "", "d", "e"]
        );
    }
}
