//! Responses events assembled from the validated chat stream.
use serde_json::{Value, json};

pub(super) struct Events {
    pub response: Value,
    output: Vec<Value>,
    closed: Vec<bool>,
    current: Option<usize>,
    tools: Vec<usize>,
    sequence: u64,
    finished: bool,
    usage: bool,
}
impl Events {
    pub fn new(response: Value) -> Self {
        Self {
            response,
            output: Vec::new(),
            closed: Vec::new(),
            current: None,
            tools: Vec::new(),
            sequence: 0,
            finished: false,
            usage: false,
        }
    }
    fn event(&mut self, kind: &str, mut value: Value) -> Value {
        value["type"] = json!(kind);
        value["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        value
    }
    pub fn start(&mut self) -> Vec<Value> {
        self.response["status"] = json!("in_progress");
        vec![
            self.event("response.created", json!({"response":self.response})),
            self.event("response.in_progress", json!({"response":self.response})),
        ]
    }
    fn base(&self, index: usize) -> Value {
        json!({"output_index":index, "item_id":self.output[index]["id"], "content_index":0})
    }
    fn close(&mut self, index: usize, events: &mut Vec<Value>) {
        if self.closed[index] {
            return;
        }
        self.closed[index] = true;
        self.output[index]["status"] = json!("completed");
        let kind = self.output[index]["type"]
            .as_str()
            .expect("item type")
            .to_owned();
        let mut done = self.base(index);
        if kind == "function_call" {
            done.as_object_mut().expect("event").remove("content_index");
            done["arguments"] = self.output[index]["arguments"].clone();
            done["name"] = self.output[index]["name"].clone();
            events.push(self.event("response.function_call_arguments.done", done));
        } else {
            let part = self.output[index]["content"][0].clone();
            done["text"] = part["text"].clone();
            let (text, part_kind) = if kind == "message" {
                done["logprobs"] = part["logprobs"].clone();
                ("response.output_text.done", "response.content_part.done")
            } else {
                (
                    "response.reasoning_text.done",
                    "response.reasoning_part.done",
                )
            };
            events.push(self.event(text, done));
            let mut done = self.base(index);
            done["part"] = part;
            events.push(self.event(part_kind, done));
        }
        events.push(self.event(
            "response.output_item.done",
            json!({"output_index":index, "item":self.output[index]}),
        ));
    }
    fn text(&mut self, kind: &str, text: &str, logprobs: &Value, events: &mut Vec<Value>) {
        if text.is_empty() && logprobs.as_array().is_none_or(Vec::is_empty) {
            return;
        }
        let index = if let Some(index) = self
            .current
            .filter(|index| self.output[*index]["type"] == kind)
        {
            index
        } else {
            if let Some(index) = self.current.take() {
                self.close(index, events);
            }
            for index in self.tools.clone() {
                self.close(index, events);
            }
            let index = self.output.len();
            let id = format!(
                "{}_{}",
                self.response["id"].as_str().expect("response id"),
                index
            );
            let mut item = if kind == "message" {
                json!({"id":id,"type":"message","role":"assistant","content":[],"status":"in_progress"})
            } else {
                json!({"id":id,"type":"reasoning","summary":[],"status":"in_progress"})
            };
            events.push(self.event(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            ));
            let part = if kind == "message" {
                json!({"type":"output_text","text":"","annotations":[],"logprobs":[]})
            } else {
                json!({"type":"reasoning_text","text":""})
            };
            item["content"] = json!([part]);
            self.output.push(item);
            self.closed.push(false);
            let mut added = self.base(index);
            added["part"] = part;
            events.push(self.event(
                if kind == "message" {
                    "response.content_part.added"
                } else {
                    "response.reasoning_part.added"
                },
                added,
            ));
            self.current = Some(index);
            index
        };
        let part = &mut self.output[index]["content"][0];
        if let Value::String(accumulated) = &mut part["text"] {
            accumulated.push_str(text);
        }
        if kind == "message"
            && let Some(rows) = logprobs.as_array()
        {
            part["logprobs"]
                .as_array_mut()
                .expect("logprobs")
                .extend(rows.clone());
        }
        let mut delta = self.base(index);
        delta["delta"] = json!(text);
        if kind == "message" {
            delta["logprobs"] = logprobs.clone();
        }
        events.push(self.event(
            if kind == "message" {
                "response.output_text.delta"
            } else {
                "response.reasoning_text.delta"
            },
            delta,
        ));
    }
    pub fn chunk(
        &mut self,
        chunk: Value,
        reasoning_field: &str,
    ) -> Result<Vec<Value>, crate::error::ApiError> {
        let mut events = Vec::new();
        if chunk["usage"].is_object() {
            let usage = &chunk["usage"];
            self.response["usage"] = json!({"input_tokens":usage["prompt_tokens"], "output_tokens":usage["completion_tokens"], "total_tokens":usage["total_tokens"],
                "input_tokens_details":{"cached_tokens":usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0)},
                "output_tokens_details":{"reasoning_tokens":usage["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0)}});
            self.usage = true;
        }
        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];
        let logprobs = choice["logprobs"]["content"].as_array().map_or_else(|| json!([]), |rows| json!(rows.iter().map(|row| json!({"token":row["token"],"logprob":row["logprob"],"top_logprobs":row["top_logprobs"]})).collect::<Vec<_>>()));
        if let Some(text) = delta[reasoning_field].as_str() {
            self.text("reasoning", text, &json!([]), &mut events);
        }
        if delta.get(reasoning_field).is_none() && delta.get("tool_calls").is_none() {
            self.text(
                "message",
                delta["content"].as_str().unwrap_or(""),
                &logprobs,
                &mut events,
            );
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            if let Some(index) = self.current.take() {
                self.close(index, &mut events);
            }
            for call in calls {
                let call_index = call["index"].as_u64().expect("validated tool index") as usize;
                if call["function"]["name"].is_string() {
                    if let Some(previous) = self.tools.last().copied() {
                        self.close(previous, &mut events);
                    }
                    let index = self.output.len();
                    let item = json!({"type":"function_call", "id":format!("{}_{}", self.response["id"].as_str().expect("id"),index), "call_id":call["id"], "name":call["function"]["name"], "arguments":"", "status":"in_progress"});
                    self.output.push(item.clone());
                    self.closed.push(false);
                    self.tools.push(index);
                    events.push(self.event(
                        "response.output_item.added",
                        json!({"output_index":index,"item":item}),
                    ));
                }
                let index = self.tools[call_index];
                if self.closed[index] {
                    return Err(crate::error::ApiError::backend(
                        "interleaved tool argument streams are not supported",
                        None,
                    ));
                }
                let arguments = call["function"]["arguments"].as_str().unwrap_or("");
                if let Value::String(accumulated) = &mut self.output[index]["arguments"] {
                    accumulated.push_str(arguments);
                }
                let mut delta = self.base(index);
                delta
                    .as_object_mut()
                    .expect("event")
                    .remove("content_index");
                delta["delta"] = json!(arguments);
                events.push(self.event("response.function_call_arguments.delta", delta));
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finished = true;
            self.response["status"] = json!(if reason == "length" {
                "incomplete"
            } else {
                "completed"
            });
            if reason == "length" {
                self.response["incomplete_details"] = json!({"reason":"max_output_tokens"});
            }
            for index in 0..self.output.len() {
                self.close(index, &mut events);
            }
        }
        Ok(events)
    }
    pub fn complete(&mut self) -> Result<Value, crate::error::ApiError> {
        if !self.finished || !self.usage {
            return Err(crate::error::ApiError::backend(
                "chat stream ended without a terminal response and usage",
                None,
            ));
        }
        self.response["output"] = json!(self.output);
        Ok(self.event("response.completed", json!({"response":self.response})))
    }
    pub fn error(&mut self, error: &crate::error::ApiError) -> Value {
        self.response["status"] = json!("failed");
        let code = match error.status {
            axum::http::StatusCode::TOO_MANY_REQUESTS => "rate_limit_exceeded",
            axum::http::StatusCode::BAD_REQUEST => "invalid_prompt",
            _ => "server_error",
        };
        self.response["error"] = json!({"code":code,"message":error.message});
        self.event(
            "error",
            json!({"code":error.kind,"message":error.message,"param":error.parameter}),
        )
    }
}
